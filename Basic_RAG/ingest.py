from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_chroma import Chroma
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"
from dotenv import load_dotenv
load_dotenv()


#Loading the document - type: doc object
loader = PyPDFLoader(r"D:\Langchain_bot1\Basic_RAG\pdfs\GIS_notes.pdf")
docs = loader.load()

#Creating chunks - type: doc object
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,   
)

chunks=splitter.split_documents(docs) 

#Sending the chunks to vector store for embedding and metadata storage
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(DB_PATH),
    collection_name="gis_notes"
)


