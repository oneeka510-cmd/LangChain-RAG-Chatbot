from dotenv import load_dotenv

from rag.config import Settings
from rag.ingestion import build_index


if __name__ == "__main__":
    load_dotenv()
    stats = build_index(Settings())
    print(f"Indexed {stats['documents']} document units into {stats['chunks']} chunks.")

