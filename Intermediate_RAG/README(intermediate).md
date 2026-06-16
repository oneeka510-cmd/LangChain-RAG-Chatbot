# Intermediate RAG Chatbot using LangChain
![Intermediate version RAG chat screenshot](image.png)

## Overview

This project is an intermediate Retrieval-Augmented Generation (RAG) chatbot built using LangChain, ChromaDB, Hugging Face Embeddings, and a Hugging Face LLM.

Unlike a basic RAG implementation that works on a single PDF, this chatbot can ingest and retrieve information from an entire directory of PDF documents. It uses semantic search, Maximum Marginal Relevance (MMR) retrieval, source citations, and a persistent vector database to provide grounded answers.

---

## Features

### Multi-PDF Knowledge Base

* Loads all PDF files from a directory.
* Builds a unified searchable knowledge base.

### Semantic Search

* Uses sentence-transformer embeddings:

  * `sentence-transformers/all-MiniLM-L6-v2`
* Retrieves relevant chunks using vector similarity.

### MMR Retrieval

* Uses LangChain retrievers with:

  * `search_type="mmr"`
* Improves diversity of retrieved chunks and reduces redundancy.

### Source Citations

* Displays source document names and page numbers.
* Helps users verify answers against original documents.

### Persistent Chroma Database

* Embeddings are generated once during ingestion.
* ChromaDB is stored locally and reused across sessions.

### LCEL Chains

* Uses LangChain Expression Language (LCEL):

```text
PromptTemplate
      ↓
Chat Model
      ↓
Output Parser
```

### Retrieval Debug Mode

* Optional debug mode for inspecting:

  * Retrieved documents
  * Metadata
  * Retrieved chunk content

---

## Architecture

```text
PDF Directory
      ↓
DirectoryLoader
      ↓
Document Chunks
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Retriever (MMR)
      ↓
Context Creation
      ↓
Prompt Template
      ↓
LLM
      ↓
Answer + Citations
```

---

## Project Structure

```text
Intermediate_RAG/

├── int_ingest.py
├── int_chat.py
├── int_prompt.py
├── pdf_folder/
├── int_chroma_db/
├── .env
└── README.md
```

---

## Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Hugging Face Inference Endpoint
* PyPDFLoader
* DirectoryLoader

---

## Retrieval Flow

### 1. User asks a question

```text
What is GIS?
```

### 2. Retriever searches vector database

```python
retriever.invoke(question)
```

### 3. Relevant chunks are retrieved

```text
Chunk 1
Chunk 2
Chunk 3
```

### 4. Context is constructed

```python
context += doc.page_content
```

### 5. Prompt is sent to the LLM

```python
chain.invoke(...)
```

### 6. Response is generated

```text
GIS is a system used for storing, analyzing,
and visualizing geographic information.
```

### 7. Sources are displayed

```text
Sources:
GIS_notes.pdf | Page 3
GIS_Assignment.pdf | Page 1
```

---

## Example

```text
ENTER YOUR QUESTION:

What is GIS?

GIS (Geographic Information System) is a system used for
capturing, storing, managing, analyzing, and visualizing
geographic data.

Sources:
GIS_notes.pdf | Page 3
GIS_notes.pdf | Page 4
```

---

## Learning Outcomes

Through this project I learned:

* Multi-document ingestion
* Document chunking strategies
* Vector databases
* Embeddings
* Semantic search
* Retriever abstraction
* Maximum Marginal Relevance (MMR)
* Prompt engineering for RAG
* Source citation handling
* LCEL chains
* Retrieval debugging techniques

---

## Future Improvements

* Retrieval confidence scoring
* Score-threshold filtering
* Streamlit UI
* Conversational memory
* Hybrid Search (Keyword + Vector Search)
* Reranking
* Full LCEL retrieval chains

---

## Key Takeaway

A good RAG system is not only about generating answers. It is also about retrieving the correct information, grounding responses in source documents, and providing transparency through citations.

