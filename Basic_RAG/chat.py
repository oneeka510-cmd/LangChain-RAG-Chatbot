from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from prompts import CHAT_PROMPT
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent #points to the folder containing the script.
DB_PATH = BASE_DIR / "chroma_db"

print(DB_PATH)
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#Just Connecting to existing db that is why not using Chroma.from_documents(...)
vectorstore = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embedding_model,
    collection_name="gis_notes"
)

# Chat model 
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

# Model object
chat_model=ChatHuggingFace(llm=llm)

parser= StrOutputParser() #just extracting content from ai msg object
chain= CHAT_PROMPT| chat_model| parser

#CHATTING LOGIC

while True:
     question=input("Enter your question \n")

     if(question.lower()=="exit"):
        break
     else:
        results = vectorstore.similarity_search(
            question,
            k=3
    )
        context=""
        for ref in results:
            context= context+ ref.page_content +"\n\n"
        
        # prompt = CHAT_PROMPT.invoke(
        # {
        # "context": context,
        # "question": question
        #  }
        #  )


        # answer=chat_model.invoke(prompt)
        # print(answer.content)
        
        
        answer= chain.invoke({
            "context": context,
            "question": question
        }
        )

        print(answer+"\n")
     

