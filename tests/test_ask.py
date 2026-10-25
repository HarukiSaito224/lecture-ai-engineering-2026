from types import SimpleNamespace
from unittest.mock import Mock

from app import ask as ask_module


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
    )

    class FakeClient:
        def __init__(self, api_key):
            self.models = SimpleNamespace(generate_content=lambda **kwargs: response)

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(
        ask_module.time,
        "perf_counter",
        Mock(side_effect=[10.0, 12.345]),
    )

    assert ask_module.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "[応答時間] 2.35 秒" in output
    assert "[トークン数] 入力: 12, 出力: 34" in output


def test_ask_displays_unknown_counts_when_usage_metadata_is_missing(
    monkeypatch, capsys
):
    response = SimpleNamespace(text="回答", usage_metadata=None)

    class FakeClient:
        def __init__(self, api_key):
            self.models = SimpleNamespace(generate_content=lambda **kwargs: response)

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)

    assert ask_module.ask("質問") == "回答"
    assert "[トークン数] 入力: 不明, 出力: 不明" in capsys.readouterr().out


def test_chat_sends_multiple_questions_through_the_same_session(monkeypatch, capsys):
    sent_questions = []
    responses = iter(
        [
            SimpleNamespace(text="最初の回答", usage_metadata=None),
            SimpleNamespace(text="続きの回答", usage_metadata=None),
        ]
    )

    class FakeChat:
        def send_message(self, question):
            sent_questions.append(question)
            return next(responses)

    class FakeClient:
        def __init__(self, api_key):
            self.chats = SimpleNamespace(create=lambda **kwargs: FakeChat())
            self.closed = False

        def close(self):
            self.closed = True

    client = FakeClient("test-key")
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        ask_module.time,
        "perf_counter",
        Mock(side_effect=[10.0, 11.0, 12.0, 14.0]),
    )
    questions = iter(["前の質問", "その続き", "/exit"])

    ask_module.chat(input_fn=lambda prompt: next(questions))

    output = capsys.readouterr().out
    assert sent_questions == ["前の質問", "その続き"]
    assert "最初の回答" in output
    assert "続きの回答" in output
    assert output.count("[応答時間]") == 2
    assert client.closed


def test_chat_can_start_with_a_question_and_exit_on_eof(monkeypatch, capsys):
    sent_questions = []

    class FakeChat:
        def send_message(self, question):
            sent_questions.append(question)
            return SimpleNamespace(text="回答", usage_metadata=None)

    class FakeClient:
        def __init__(self, api_key):
            self.chats = SimpleNamespace(create=lambda **kwargs: FakeChat())

        def close(self):
            pass

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    ticks = iter([1.0, 2.0])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(ticks))

    def input_until_eof(prompt):
        raise EOFError

    ask_module.chat(initial_question="最初の質問", input_fn=input_until_eof)

    assert sent_questions == ["最初の質問"]
    assert "回答" in capsys.readouterr().out
