from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)


# Model object
chat_model=ChatHuggingFace(llm=llm)

# result=chat_model.invoke("")
# print(result.content)




#PDF Loading and displaying
loader = PyPDFLoader(r"D:\Langchain_bot1\Basic_RAG\pdfs\GIS_notes.pdf")
docs = loader.load()

# print("Pages:", len(docs)) 
# num_pages=len(docs)

# # print(docs[0].page_content[:500])
# print(docs[0].page_content)

# pdf_content= " "  #This will store all the content from pdf in text format

# #there are 12 pages, now I am making a string of all document objects' page_content

# for i in range(0,num_pages):
#     pdf_content= pdf_content+ docs[i].page_content

#There is no need to convert the content to a single string coz now each page is a document object and it preserves metadata which can ;ater be used for referencing, and langchain splitters work on document objects as well
# print(type(docs))
# print(type(docs[0]))
# print(docs[0].metadata)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,   
)

chunks=splitter.split_documents(docs) 
#chunks are also a list of document objects

# print(type(chunks[0]))
# print(chunks[0].metadata)
# print(chunks[0].page_content[:300])


#Up untill this point we have 
# PDF
# ↓
# Loader
# ↓
# Documents
# ↓
# Chunking
# ↓
# Chunk Documents

#Analysis:
# print("Number of pages:", len(docs))
# print("Number of chunks:", len(chunks))

# print("\nChunk Metadata:")
# print(chunks[0].metadata)

# print("\nChunk Length:")
# print(len(chunks[0].page_content))

# print("\nChunk Content:")
# print(chunks[0].page_content)

# print("Chunks:", len(chunks))

# for i in range(3):
#     print(f"\n----- CHUNK {i} -----")
#     print(chunks[i].page_content)



#Creating embeddings now



embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # embed_documents() expects: List[str] not List[Document]

# texts = [chunk.page_content for chunk in chunks]

# embedding_result = embedding.embed_documents(texts)

# print(len(embedding_result))
# print(len(embedding_result[0]))


vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

#print(vectorstore._collection.count())

question= "What are the applications of GIS?"

results = vectorstore.similarity_search(
    question,
    k=3
)

# print(results[0].page_content)


context= ""

for ref in results:
    context= context + ref.page_content+"\n\n"

# print(context)
# print(type(context)) # seeing because llms work with text not document objects


prompt = f"""
Answer the question briefly using only the provided context.

Context:
{context}
Question:
{question}
"""

# print(prompt)


output=chat_model.invoke(prompt)
print(output.content)



