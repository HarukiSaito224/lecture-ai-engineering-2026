"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って対話する

ツールは持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time
from collections.abc import Callable

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed = time.perf_counter() - started_at

    print_response_stats(response, elapsed)
    return response.text


def print_response_stats(response, elapsed: float) -> None:
    print(f"[応答時間] {elapsed:.2f} 秒")
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(
        f"[トークン数] 入力: {input_tokens if input_tokens is not None else '不明'}, "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )


def chat(
    initial_question: str | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> None:
    """同じ Gemini チャットセッションで質問を続け、会話履歴を引き継ぐ。"""
    client = genai.Client(api_key=get_api_key())
    try:
        session = client.chats.create(model=MODEL)
        read_input = input if input_fn is None else input_fn
        question = initial_question
        while True:
            if question is None:
                try:
                    question = read_input("あなた> ")
                except EOFError:
                    print()
                    break

            if question.strip().lower() in {"/exit", "/quit"}:
                break
            if not question.strip():
                question = None
                continue

            print(f"[質問] {question}")
            started_at = time.perf_counter()
            response = session.send_message(question)
            elapsed = time.perf_counter() - started_at
            print(response.text)
            print_response_stats(response, elapsed)
            question = None
    except KeyboardInterrupt:
        print("\nチャットを終了します。")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保って対話する（終了: /exit または /quit）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(initial_question=args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
