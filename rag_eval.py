"""
RAG 评测：用同一套题，分别量出「检索」和「回答」得了多少分。

用法：
    python rag_eval.py docs --no-llm              # 只测检索（BM25），不花钱
    python rag_eval.py docs --no-llm --vector     # 只测检索（向量）
    python rag_eval.py docs --vector              # 检索 + 生成 + 打分

结果按检索方式分别写入 eval_result_bm25.json / eval_result_vector.json，
方便两种方案直接对比。

题目和参考答案在 eval_set.json。改题就改那个文件，别改这个脚本。
"""

import json
import re
import sys
import time
from pathlib import Path

from rag_step1 import BM25, build_corpus, tokenize
from rag_step2 import BASE_URL, MODEL, SYSTEM_PROMPT, build_prompt, load_key

JUDGE_SYSTEM = """你是一个严格但公正的评分员。请比较【参考答案】和【模型回答】。

评分标准：
2 = 正确：核心信息与参考答案一致。
1 = 部分正确：答对了一部分，但遗漏关键信息，或有小错误。
0 = 错误：与参考答案矛盾、答非所问，或资料里没有却编造了答案。

只输出一行 JSON，不要解释：
{"score": 0, "reason": "一句话理由"}
"""


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def check_hit(hits: list[dict], gold: str | None) -> int | None:
    """gold 出现在第几名（从 1 开始）。None 表示这题不考检索，0 表示没命中。"""
    if gold is None:
        return None
    g = norm(gold)
    for rank, c in enumerate(hits, 1):
        if g in norm(c["text"]):
            return rank
    return 0


def generate(client, question: str, hits: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def judge(client, item: dict, answer: str) -> tuple[int, str]:
    user = (
        f"【问题】{item['question']}\n"
        f"【参考答案】{item['answer']}\n"
        f"【模型回答】{answer}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return 0, f"评分输出无法解析：{text[:60]}"
    try:
        data = json.loads(m.group())
        return int(data.get("score", 0)), str(data.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        return 0, f"评分输出无法解析：{text[:60]}"


def main() -> None:
    argv = sys.argv[1:]
    corpus_arg, i = None, 0
    while i < len(argv):
        if argv[i] in {"--k", "--eval"}:
            i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            corpus_arg, i = argv[i], i + 1
    corpus = Path(corpus_arg) if corpus_arg else Path("docs")
    use_llm = "--no-llm" not in sys.argv
    k = 3
    if "--k" in sys.argv:
        k = int(sys.argv[sys.argv.index("--k") + 1])

    if not Path("eval_set.json").exists():
        print("找不到 eval_set.json")
        return

    items = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))
    chunks = build_corpus(corpus)
    if not chunks:
        print(f"{corpus} 里没有可用内容")
        return

    retriever = "bm25"
    if "--vector" in sys.argv:
        retriever = "vector"
    elif "--hybrid" in sys.argv:
        retriever = "hybrid"

    if retriever == "hybrid":
        from rag_step3 import HybridIndex

        index = HybridIndex(chunks)
    elif retriever == "vector":
        from rag_step3 import VectorIndex

        index = VectorIndex(chunks)
    else:
        index = BM25([tokenize(c["text"]) for c in chunks])

    client = None
    if use_llm:
        key = load_key()
        if not key:
            print("没读到 DEEPSEEK_API_KEY，先跑 --no-llm，或在 .env 里配置。")
            return
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=BASE_URL)

    print(f"语料：{corpus}   共 {len(chunks)} 块   检索：{retriever}   题目：{len(items)} 道\n")
    print(f"{'问题':<28} {'检索':<8} {'生成':<6} 理由")
    print("-" * 78)

    rows = []
    for n, item in enumerate(items, 1):
        hits = [chunks[i] for _, i in index.search(item["question"], k)]
        rank = check_hit(hits, item.get("gold"))
        rank_txt = "不适用" if rank is None else ("第1" if rank == 1 else ("第%d" % rank if rank else "未命中"))

        score, reason, answer = None, "", ""
        if use_llm and hits:
            print(f"  [{n}/{len(items)}] 生成中…", end="\r")
            answer = generate(client, item["question"], hits)
            score, reason = judge(client, item, answer)
        elif use_llm and not hits:
            answer, reason = "（检索为空，直接拒答）", "没有检索到资料，未调用模型"

        rows.append(
            {
                "question": item["question"],
                "gold": item.get("gold"),
                "rank": rank,
                "score": score,
                "reason": reason,
                "answer": answer,
            }
        )
        score_txt = "-" if score is None else f"{score}/2"
        print(f"{item['question']:<28} {rank_txt:<8} {score_txt:<6} {reason[:40]}")

    # ---------------- 汇总
    scored = [r for r in rows if r["rank"] is not None]
    hit1 = sum(1 for r in scored if r["rank"] == 1)
    hitk = sum(1 for r in scored if r["rank"] and r["rank"] <= k)
    mrr = sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored) if scored else 0

    print("\n" + "=" * 78)
    if scored:
        print(f"检索：top-1 命中 {hit1}/{len(scored)}   top-{k} 命中 {hitk}/{len(scored)}   MRR {mrr:.2f}")
    graded = [r for r in rows if r["score"] is not None]
    if graded:
        avg = sum(r["score"] for r in graded) / len(graded)
        print(f"生成：平均 {avg:.2f}/2（{len(graded)} 题）")

    out = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": str(corpus),
        "retriever": retriever,
        "chunks": len(chunks),
        "k": k,
        "hit1": hit1 if scored else None,
        "hitk": hitk if scored else None,
        "mrr": round(mrr, 3) if scored else None,
        "items": rows,
    }
    out_file = f"eval_result_{retriever}{'' if k == 3 else f'_k{k}'}.json"
    Path(out_file).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out_file}")


if __name__ == "__main__":
    main()
