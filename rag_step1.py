"""
RAG 第一步：只做「检索」。

不调用大模型、不装任何第三方库、不需要 API key。
目的：亲眼看见「提问 -> 切块 -> 打分 -> 返回最相关的几段」这个过程。

用法：
    python rag_step1.py                      # 默认检索 README.md
    python rag_step1.py docs                 # 检索整个 docs 文件夹（推荐）
    python rag_step1.py notes.md             # 检索单个文件
    python rag_step1.py docs --eval questions.txt   # 批量跑评测问题
    python rag_step1.py --test               # 跑内置的 5 个测试问题（针对本 README）

支持 .md / .txt / .py；装了 pypdf 之后也支持 .pdf。
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- 1. 读文件

SUPPORTED = {".md", ".txt", ".py", ".pdf"}
SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules", ".git", "dist", "build"}
SKIP_FILES = {"questions.txt"}   # 评测题不要混进语料，否则会自己命中自己


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in SUPPORTED:
            continue
        if f.name.lower() in SKIP_FILES:
            continue
        if any(p in SKIP_DIRS or p.startswith(".") for p in f.parts):
            continue
        yield f


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"[跳过] {path.name}：读 PDF 需要先 pip install pypdf")
            return ""
        return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    # 有些 Windows 记事本存的文件是 GBK，先按 UTF-8 读，失败再退回 GBK
    # utf-8-sig 顺便去掉 BOM，不然第一块会带着一个不可见字符
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- 2. 切块

CHUNK_SIZE = 350      # 每块大约多少字
CHUNK_OVERLAP = 80    # 相邻块重叠多少字，避免答案正好被切断


def split_document(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题和空行切段，返回 [(标题, 段落), ...]。"""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    heading = ""

    def flush() -> None:
        block = "\n".join(buf).strip()
        if block:
            out.append((heading, block))
        buf.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
        elif line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return out


def chunk_block(block: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """太长的段落再按字数切开，保留一点重叠。"""
    if len(block) <= size:
        return [block]
    pieces, start = [], 0
    while start < len(block):
        pieces.append(block[start : start + size])
        start += max(1, size - overlap)
    return pieces


def build_corpus(
    path: Path,
    exclude: set[Path] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    exclude = exclude or set()
    chunks: list[dict] = []
    for f in iter_files(path):
        if f.resolve() in exclude:
            continue
        text = read_file(f)
        if not text.strip():
            continue
        for heading, block in split_document(text):
            for piece in chunk_block(block, chunk_size, overlap):
                chunks.append(
                    {
                        "source": f.name,
                        "heading": heading,
                        "text": f"{heading}\n{piece}" if heading else piece,
                    }
                )
    return chunks


# ---------------------------------------------------------------- 3. 分词

LATIN_RE = re.compile(r"[a-zA-Z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中文按「单字 + 相邻两字」切开，英文按单词切开。

    中文没有空格，所以用 bigram 代替词：'启动服务' -> '启动' '动服' '服务'
    这样做很土，但对关键词检索意外地好用。
    """
    text = text.lower()
    tokens = LATIN_RE.findall(text)
    for seg in CJK_RE.findall(text):
        tokens.extend(seg)                                           # 单字
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))   # 两字
    return tokens


# ---------------------------------------------------------------- 4. 打分（BM25）

class BM25:
    """经典的检索打分算法，本质是「这个词在这块里出现得多、在别处少见，就更相关」。"""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avg_len = sum(len(d) for d in docs) / len(docs)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        total = 0.0
        for term in query:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
            )
        return total

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, int]]:
        q = tokenize(query)
        scored = [(self.score(q, i), i) for i in range(len(self.docs))]
        scored.sort(reverse=True)
        return [x for x in scored[:top_k] if x[0] > 0]


# ---------------------------------------------------------------- 5. 主程序

def show(query: str, chunks: list[dict], index: BM25, top_k: int = 3) -> list[dict]:
    hits = index.search(query, top_k)
    print(f"\n问题：{query}")
    if not hits:
        q = tokenize(query)
        known = [t for t in dict.fromkeys(q) if t in index.idf]
        print(f"  语料共 {len(chunks)} 块，但没有一块包含你问题里的词。")
        print(f"  问题被切成：{' '.join(dict.fromkeys(q))[:80]}")
        if not known:
            print("  这些词在文档里一次都没出现过。BM25 只认字面，换一两个说法再试；")
            print("  如果换几种说法都不行，那就不是你的问题——该上向量检索了（它认意思，不认字）。")
        else:
            print(f"  其中 {len(known)} 个词出现过，但没有集中出现在同一块里。")
        print("  想看语料里到底有什么：命令后面加 --stats")
        return []
    for rank, (score, i) in enumerate(hits, 1):
        c = chunks[i]
        preview = c["text"].replace("\n", " / ")
        print(f"\n  [{rank}] 得分 {score:5.2f}  {c['source']}")
        print(f"      {preview[:150]}...")
    return [chunks[i] for _, i in hits]


TESTS = [
    ("每学期最多能选多少学分？", "30 学分"),
    ("选课分几轮？", "两轮"),
    ("请假超过一天找谁审批？", "学院审批"),
    ("考试作弊会怎么处理？", "记零分"),
    ("图书馆一次最多借几本？", "10 本"),
]


def print_stats(chunks: list[dict], corpus: Path) -> None:
    """看语料到底读进来了什么。检索不出结果时，第一步永远是查这个。"""
    print(f"\n语料：{corpus}")
    print(f"共 {len(chunks)} 块：")
    for source, n in Counter(c["source"] for c in chunks).most_common():
        print(f"  {source:<40} {n:>4} 块")
    print(f"\n每块开头（最多 15 块，只看前 70 字）：")
    for c in chunks[:15]:
        body = " ".join(c["text"].split())[:70]
        print(f"  [{c['source']}] {c['heading'] or '—'}  |  {body}")


def eval_path_from_args() -> Path | None:
    if "--eval" in sys.argv:
        i = sys.argv.index("--eval")
        if i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return None


def run_eval(path: Path, chunks: list[dict], index: BM25) -> None:
    questions = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"共 {len(questions)} 个问题。看每题的 top-3 里有没有正确答案，自己数命中数。\n")
    for q in questions:
        show(q, chunks, index)
        print("\n  " + "-" * 60)


def main() -> None:
    argv = sys.argv[1:]
    corpus_arg, i = None, 0
    while i < len(argv):
        if argv[i] == "--eval":
            i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            corpus_arg, i = argv[i], i + 1
    corpus = Path(corpus_arg) if corpus_arg else Path("README.md")

    if not corpus.exists():
        print(f"找不到 {corpus}，先建一个文件夹把资料放进去。")
        return

    eval_file = eval_path_from_args()
    chunks = build_corpus(corpus, exclude={eval_file.resolve()} if eval_file else set())
    if not chunks:
        print(f"{corpus} 里没有可用内容（支持 {'/'.join(sorted(SUPPORTED))}）。")
        return

    index = BM25([tokenize(c["text"]) for c in chunks])
    print(f"语料：{corpus}   共 {len(chunks)} 块，来自 {len({c['source'] for c in chunks})} 个文件")

    if "--stats" in sys.argv:
        print_stats(chunks, corpus)
        return

    if "--test" in sys.argv:
        hit = 0
        for q, expected in TESTS:
            tops = show(q, chunks, index)
            ok = any(expected in t["text"] for t in tops)
            hit += ok
            print(f"  => {'命中' if ok else f'未命中（期望段落含“{expected}”）'}")
        print(f"\n合计：{hit}/{len(TESTS)}")
        return

    if eval_file:
        if not eval_file.exists():
            print(f"找不到问题文件 {eval_file}")
            return
        run_eval(eval_file, chunks, index)
        return

    print("输入问题后回车，输入 q 退出。")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q:
            show(q, chunks, index)


if __name__ == "__main__":
    main()
