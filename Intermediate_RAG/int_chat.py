from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from int_prompt import CHAT_PROMPT
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "int_chroma_db"

print(DB_PATH)
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#Just Connecting to existing db that is why not using Chroma.from_documents(...)
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=str(DB_PATH),
    collection_name="int_sample"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
#Retriever = first-class component in LCEL chains

# Chat model 
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

# Model object
chat_model=ChatHuggingFace(llm=llm)



parser= StrOutputParser() 
chain= CHAT_PROMPT| chat_model| parser


#CHATTING LOGIC

while True:
     question=input("\n\nENTER YOUR QUESTION (To end conversation type - exit)  :  ")
    
     if(question.lower()=="exit"):
        break
     else:
        results = retriever.invoke(question)
        #retrieval debugging mode- for checking retrieval quality
        DEBUG = False
        if DEBUG:
            print("\nRetrieved Documents:\n")
            for docu in results:
                print(docu.metadata)
                print(docu.page_content[:400])

        sources=set()
        context=""
        for ref in results:
              context= context+ ref.page_content+ "\n\n"
            
              sources.add(
                # f"{ref.metadata['source']} | Page {ref.metadata['page_label']}"
                f"{os.path.basename(ref.metadata['source'])} | Page {ref.metadata['page_label']}"
                 )
    
        answer= chain.invoke({
            "context": context,
            "question": question
        }
        )
        
        if "could not find relevant information" in answer.lower():
             print(answer)
        else:
             print(answer)
             print("\nSources:")
             for i in sources:
                print(i)

#   Retriever
#   ↓
#   LLM
#   ↓
#   String Matching


# But eventually we want this:

#   Retriever
#   ↓
#   Confidence Check
#   ↓
#   LLM

