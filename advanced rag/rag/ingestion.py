import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata["source"] = path.name
            document.metadata["path"] = str(path.relative_to(folder))
            document.metadata["page"] = int(document.metadata.get("page", 0))
        documents.extend(loaded)
    if not documents:
        raise ValueError(f"No PDF, Markdown, or text documents found in {folder}")
    return documents


def build_index(settings: Settings, reset: bool = True) -> dict[str, int]:
    documents = load_documents(settings.documents_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    if reset and settings.db_dir.exists():
        shutil.rmtree(settings.db_dir)
    settings.db_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.db_dir),
        collection_name=settings.collection_name,
    )
    payload = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    settings.chunks_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"documents": len(documents), "chunks": len(chunks)}


def read_cached_chunks(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError("Index not found. Run `python ingest.py` first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Document(**item) for item in payload]

