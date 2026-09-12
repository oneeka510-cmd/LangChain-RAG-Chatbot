from contextlib import asynccontextmanager
from threading import Lock
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import ROOT, Settings
from rag.ingestion import build_index
from rag.memory import SummarizingMemory
from rag.service import AdvancedRAG


load_dotenv()
STATIC_DIR = ROOT / "frontend"
settings = Settings()
state: dict = {"rag": None, "error": None, "sessions": {}}
state_lock = Lock()


def load_rag() -> None:
    try:
        state["rag"] = AdvancedRAG(settings)
        state["error"] = None
    except Exception as exc:
        state["rag"] = None
        state["error"] = str(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.chunks_file.exists():
        load_rag()
    yield


app = FastAPI(title="Advanced RAG API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=80)
    threshold: float | None = Field(default=None, ge=0, le=1)


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {
        "ready": state["rag"] is not None,
        "error": state["error"],
        "indexed": settings.chunks_file.exists(),
        "reranking": settings.rerank_enabled,
        "default_threshold": settings.confidence_threshold,
    }


@app.post("/api/reindex")
def reindex():
    with state_lock:
        stats = build_index(settings)
        load_rag()
    if state["rag"] is None:
        raise HTTPException(status_code=500, detail=state["error"])
    return stats


@app.post("/api/chat")
def chat(request: ChatRequest):
    rag = state["rag"]
    if rag is None:
        raise HTTPException(status_code=503, detail=state["error"] or "Build the index first.")
    memory = state["sessions"].setdefault(request.session_id, SummarizingMemory())
    try:
        return rag.ask(request.question.strip(), threshold=request.threshold, memory=memory)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/clear")
def clear_session(request: SessionRequest):
    state["sessions"].pop(request.session_id, None)
    return {"cleared": True}

