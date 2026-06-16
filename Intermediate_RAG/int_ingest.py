from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import os

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "int_chroma_db"


#Loading the document - type: doc object
loader= DirectoryLoader(
    path= r"D:\Langchain_bot1\Intermediate_RAG\pdf_folder",
    glob= '*.pdf',
    loader_cls=PyPDFLoader
)
docs=loader.load()


#Creating chunks - type: doc object
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,   
)
chunks=splitter.split_documents(docs) 


#Sending the chunks to vector store for embedding and metadata storage
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store= Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(DB_PATH),
    collection_name="int_sample"
)
print(vector_store._collection.count())