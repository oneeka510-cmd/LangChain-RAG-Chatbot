# PDF RAG Chatbot using LangChain
![Basic version RAG chat screenshot](<Screenshot 2026-06-14 210947.png>)
## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, Hugging Face Embeddings, and a Hugging Face LLM.

The chatbot can answer questions based on the contents of a PDF document by retrieving relevant information from a vector database and providing it as context to the language model.

---

## How It Works

### Ingestion Pipeline

1. Load PDF documents
2. Split documents into chunks
3. Generate embeddings for each chunk
4. Store embeddings and metadata in ChromaDB

### Chat Pipeline

1. Accept user question
2. Retrieve relevant chunks using semantic search
3. Build context from retrieved chunks
4. Inject context into a prompt template
5. Generate answer using the LLM

---

## Project Structure

```text
.
├── ingest.py           # Creates and stores vector embeddings
├── chat.py             # Chat interface and retrieval logic
├── prompts.py          # Prompt templates
├── OLD_main.py         # Initial monolithic implementation
├── pdfs/               # Source PDF document
├── chroma_db/          # Persistent Chroma database
├── .env
└── basic_readme.md
```

---

## Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Hugging Face LLM Endpoint
* PyPDFLoader

---

## Retrieval Pipeline

```text
PDF
 ↓
Loader
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Similarity Search
 ↓
Context
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

---

## Features

* PDF document ingestion
* Semantic search using embeddings
* Persistent Chroma vector database
* Prompt templates using LangChain
* LCEL chains
* Interactive command-line chatbot

---

## Setup

### Clone Repository

```bash
git clone <repo-url>
cd <repo-name>
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file and add:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## Create Vector Database

Run:

```bash
python ingest.py
```

This will:

* Load PDFs
* Create chunks
* Generate embeddings
* Store them in ChromaDB

---

## Start Chatbot

Run:

```bash
python chat.py
```

Example:

```text
Enter your question:
What is GIS?

Answer:
GIS stands for Geographic Information System...
```

Type:

```text
exit
```

to close the chatbot.

---

## Learning Outcomes

Through this project, I learned:

* Document loading in LangChain
* Text chunking strategies
* Embeddings and vector databases
* Semantic retrieval
* Prompt engineering
* LCEL (LangChain Expression Language)
* Building an end-to-end RAG pipeline

---

## Future Improvements

* Multiple PDF support
* Source citations
* Streamlit UI
* Retriever-based chains (Because retrievers are Runnables)
* Conversational memory
* Hybrid search

```
```



## A useful distinction:
Documents- Returned by loaders, splitters, retrievers: