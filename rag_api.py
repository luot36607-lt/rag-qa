"""
把 RAG 包成一个 HTTP 服务。

启动：
    .venv\\Scripts\\python.exe -m uvicorn rag_api:app --reload --port 8001

然后打开 http://127.0.0.1:8001/ 就是网页界面。

接口：
    GET  /              网页界面
    GET  /health        健康检查（顺便告诉你索引了多少块）
    GET  /samples       示例问题（取自 eval_set.json）
    POST /ask           提问，返回回答 + 引用来源
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_step1 import build_corpus
from rag_step2 import BASE_URL, MODEL, SYSTEM_PROMPT, build_prompt, load_key
from rag_step3 import VectorIndex

CORPUS_DIR = Path("docs")
WEB_PAGE = Path("web/index.html")
TOP_K = 3

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时把语料读进来、算好向量，之后每个请求都复用。"""
    chunks = build_corpus(CORPUS_DIR)
    if not chunks:
        raise RuntimeError(f"{CORPUS_DIR} 里没有可用内容")

    state["chunks"] = chunks
    state["index"] = VectorIndex(chunks)

    key = load_key()
    if not key:
        raise RuntimeError("没读到 DEEPSEEK_API_KEY，请检查 .env")

    from openai import OpenAI

    state["client"] = OpenAI(api_key=key, base_url=BASE_URL)
    print(f"已加载 {len(chunks)} 块语料，服务就绪")
    yield
    state.clear()


app = FastAPI(title="RAG 问答服务", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="你的问题")
    k: int = Field(TOP_K, ge=1, le=10, description="取几段资料，默认 3")


class Source(BaseModel):
    index: int          # 对应回答里的 [1] [2]
    file: str
    heading: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def home() -> FileResponse:
    if not WEB_PAGE.exists():
        raise HTTPException(status_code=404, detail=f"找不到 {WEB_PAGE}")
    return FileResponse(WEB_PAGE)


@app.get("/samples")
def samples() -> dict:
    """给页面上的示例按钮用，直接取评测集里的前三个问题。"""
    path = Path("eval_set.json")
    if not path.exists():
        return {"questions": []}
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        return {"questions": [it["question"] for it in items[:3]]}
    except (json.JSONDecodeError, KeyError):
        return {"questions": []}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(state.get("chunks", []))}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    index: VectorIndex = state["index"]
    chunks: list[dict] = state["chunks"]

    hits = [(score, chunks[i]) for score, i in index.search(req.question, req.k)]

    # 没检索到就不调用模型：省钱，也避免它凭空编
    if not hits:
        return AskResponse(answer="资料里没有提到。", sources=[])

    prompt = build_prompt(req.question, [c for _, c in hits])
    try:
        resp = state["client"].chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{e}") from e

    return AskResponse(
        answer=resp.choices[0].message.content.strip(),
        sources=[
            Source(
                index=n,
                file=c["source"],
                heading=c["heading"],
                score=round(float(score), 3),
                snippet=" ".join(c["text"].split())[:120],
            )
            for n, (score, c) in enumerate(hits, 1)
        ],
    )
