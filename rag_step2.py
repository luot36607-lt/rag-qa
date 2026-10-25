"""
RAG 第二步：把检索到的资料交给 DeepSeek，让它「看着资料回答」。

前提：
    pip install openai
    项目根目录的 .env 里写一行：DEEPSEEK_API_KEY=sk-xxx

用法：
    python rag_step2.py docs --dry     # 只打印要发给模型的 prompt，不花钱
    python rag_step2.py docs           # 真实调用
    python rag_step2.py README.md      # 换语料
"""

import os
import sys
from pathlib import Path

from rag_step1 import BM25, build_corpus, tokenize

BASE_URL = "https://api.deepseek.com"   # DeepSeek 兼容 OpenAI 的接口协议
MODEL = "deepseek-chat"
TOP_K = 3

SYSTEM_PROMPT = """你是一个严谨的文档问答助手。规则：
1. 只能依据下面给出的【资料】回答，不要使用资料以外的知识。
2. 如果资料里没有答案，直接回答「资料里没有提到」，不要猜测。
3. 回答尽量简短，并在依据的句子后面标注片段编号，如 [1]、[2]。
"""


def load_key() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("DEEPSEEK_API_KEY") or None


def build_prompt(question: str, hits: list[dict]) -> str:
    parts = []
    for i, c in enumerate(hits, 1):
        where = f"{c['source']} · {c['heading']}" if c["heading"] else c["source"]
        parts.append(f"[{i}] 来源：{where}\n{c['text']}")
    return "【资料】\n" + "\n\n".join(parts) + f"\n\n【问题】\n{question}"


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    corpus = Path(argv[0]) if argv else Path("README.md")
    dry = "--dry" in sys.argv

    if not corpus.exists():
        print(f"找不到 {corpus}")
        return

    chunks = build_corpus(corpus)
    if not chunks:
        print(f"{corpus} 里没有可用内容")
        return

    index = BM25([tokenize(c["text"]) for c in chunks])
    print(f"语料：{corpus}   共 {len(chunks)} 块")
    print("输入问题后回车，输入 q 退出。\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        hits = [chunks[i] for score, i in index.search(question, TOP_K)]

        # 检索不到就直接拒答，不要浪费一次 API 调用，也避免模型瞎编
        if not hits:
            print("（资料里没有相关内容，直接拒答）\n")
            continue

        print("\n--- 检索到的资料 ---")
        for i, c in enumerate(hits, 1):
            where = f"{c['source']} · {c['heading']}" if c["heading"] else c["source"]
            print(f"[{i}] {where}")

        prompt = build_prompt(question, hits)

        if dry:
            print("\n--- 将发送给模型的 prompt ---")
            print(prompt)
            print("\n（--dry 模式，没有调用 API）\n")
            continue

        key = load_key()
        if not key:
            print("\n没读到 DEEPSEEK_API_KEY。请在项目根目录建 .env，内容：")
            print("DEEPSEEK_API_KEY=sk-你的key\n")
            return

        try:
            from openai import OpenAI
        except ImportError:
            print("\n需要先安装：pip install openai\n")
            return

        client = OpenAI(api_key=key, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        print("\n--- 回答 ---")
        print(resp.choices[0].message.content)
        print()


if __name__ == "__main__":
    main()
