from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import os

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "int_chroma_db"

#Step 1 : document ingestion: this time: an entire directory of pdfs
loader= DirectoryLoader(
    path= r"D:\Langchain_bot1\Intermediate_RAG\pdf_folder",
    glob= '*.pdf',
    loader_cls=PyPDFLoader
)
docs=loader.load()
# print(docs[20].page_content)


#step 2 : split into chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,   
)
chunks=splitter.split_documents(docs) 
#chunks are also document objects
# print(len(chunks))
# print(chunks[0].page_content[:300])
# print(chunks[0].metadata["page_label"])
# Content → chunk.page_content  doc object
# Metadata → chunk.metadata["key"] dictionary

# step 3: Analysis:
# print("Number of pages:", len(docs))
# print("Number of chunks:", len(chunks))

#Step 4 : embed this and create vector store
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store= Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(DB_PATH),
    collection_name="int_sample"
)
# Chroma.from_documents(...) → create/populate DB
# Chroma(...) → connect to an existing DB
# Ingestion pipeline is complete till here, after this we ask questions and reply
print(vector_store._collection.count())


#step 5 : building context
question=input("Enter your question please")
ref=vector_store.similarity_search(
    query=question,
    k=3
)
# print(ref)
#building context
context=""
for i in ref:
    context= context+ i.page_content+ "(source : "+i.metadata["source"]+")"+ "(page number: "+i.metadata["page_label"]+")" "\n\n"
# print(context)


# step 6
prompt=f"""You are a polite chatbot, summarize answer from context: {context}, 
the question is {question}"""


# step 7 Chat model 
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)


model=ChatHuggingFace(llm=llm)

answer=model.invoke(prompt)

print(answer.content)




#now separate the reponsibilities and add features:
# 1 retrievable usage
# 2 conversational memory send to gpt
# 3 source citation proper by gpt answer
# 4 Use chains wherever necessary


# Multiple PDFs
# Retriever chains
# Source citations
# Streamlit
# Conversation memory