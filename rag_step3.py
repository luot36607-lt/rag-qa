"""
RAG 第三步：把检索从「关键词匹配」换成「向量相似度」。

BM25 比的是「字面像不像」，向量比的是「意思像不像」。
所以「她为什么不出声」终于能找到写着「哑巴」的那一段。

前提：
    .env 里加一行：SILICONFLOW_API_KEY=sk-xxx
    （硅基流动的 BAAI/bge-m3 是免费的，注册就能用）

用法：
    python rag_step3.py docs
    python rag_step3.py docs --eval questions.txt

第一次会把 15 个块送去算向量，之后缓存在 embed_cache.json，不再重复花钱。
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from rag_step1 import BM25, build_corpus, tokenize

CACHE_FILE = Path("embed_cache.json")
BATCH = 32


def _env(name: str, default: str) -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get(name) or default


def model_name() -> str:
    return _env("EMBED_MODEL", "BAAI/bge-m3")


def get_client():
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("EMBED_API_KEY")
    if not key:
        raise SystemExit(
            "没读到 SILICONFLOW_API_KEY。\n"
            "去 https://cloud.siliconflow.cn 注册，拿一个 key，写进 .env：\n"
            "SILICONFLOW_API_KEY=sk-xxx"
        )
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=_env("EMBED_BASE_URL", "https://api.siliconflow.cn/v1"))


def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _key(text: str) -> str:
    return hashlib.sha1(f"{model_name()}::{text}".encode()).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    cache = _load_cache()
    out: list[list[float] | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []

    for i, t in enumerate(texts):
        k = _key(t)
        if k in cache:
            out[i] = cache[k]
        else:
            todo.append((i, t))

    if todo:
        client = get_client()
        print(f"  需要新算 {len(todo)} 个向量（其余走缓存）…")
        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            try:
                resp = client.embeddings.create(
                    model=model_name(), input=[t for _, t in batch]
                )
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"调用 embedding 失败：{e}") from e
            for (i, _), d in zip(batch, resp.data):
                v = _norm(d.embedding)
                cache[_key(texts[i])] = v
                out[i] = v
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

    return [v for v in out if v is not None]


class VectorIndex:
    """和 BM25 用同一个接口：search(query, k) -> [(分数, 块下标)]。"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectors = embed_texts([c["text"] for c in chunks])

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, int]]:
        q = embed_texts([query])[0]
        sims = [(sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self.vectors)]
        sims.sort(reverse=True)
        return sims[:top_k]


class HybridIndex:
    """两路检索，各排各的名次，再按名次融合。

    RRF（倒数排名融合）：score = 1/(60+名次)。
    好处是不用管两边的分数尺度——BM25 是 0~20，向量是 0~1，本来就没法直接相加。
    """

    def __init__(self, chunks: list[dict]):
        self.bm25 = BM25([tokenize(c["text"]) for c in chunks])
        self.vector = VectorIndex(chunks)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, int]]:
        fused: dict[int, float] = {}
        for ranks in (self.bm25.search(query, top_k * 4), self.vector.search(query, top_k * 4)):
            for rank, (_, i) in enumerate(ranks, 1):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        return sorted(((s, i) for i, s in fused.items()), reverse=True)[:top_k]


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    corpus = Path(argv[0]) if argv else Path("docs")

    if not corpus.exists():
        print(f"找不到 {corpus}")
        return

    chunks = build_corpus(corpus)
    if not chunks:
        print(f"{corpus} 里没有可用内容")
        return

    print(f"语料：{corpus}   共 {len(chunks)} 块   模型：{model_name()}")
    if "--hybrid" in sys.argv:
        print("检索：BM25 + 向量（RRF 融合）")
        index = HybridIndex(chunks)
    else:
        print("检索：纯向量")
        index = VectorIndex(chunks)

    print("输入问题后回车，输入 q 退出。\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        hits = index.search(q, 3)
        print(f"\n问题：{q}")
        for rank, (score, i) in enumerate(hits, 1):
            c = chunks[i]
            preview = c["text"].replace("\n", " / ")
            print(f"\n  [{rank}] 相似度 {score:.3f}  {c['source']}")
            print(f"      {preview[:150]}...")


if __name__ == "__main__":
    main()
