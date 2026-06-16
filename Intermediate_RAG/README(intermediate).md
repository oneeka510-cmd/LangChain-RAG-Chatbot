# Intermediate RAG Chatbot using LangChain
![Intermediate version RAG chat screenshot](image.png)
![Also intermediate version showing context resolution](image.png)

## Overview

This project is an intermediate Retrieval-Augmented Generation (RAG) chatbot built using LangChain, ChromaDB, Hugging Face Embeddings, and a Hugging Face LLM.

Unlike a basic RAG implementation that works on a single PDF, this chatbot can ingest and retrieve information from an entire directory of PDF documents. It uses semantic search, Maximum Marginal Relevance (MMR) retrieval, source citations, conversational memory, and a persistent vector database to provide grounded answers.

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
* Citations are generated from retrieved metadata rather than the LLM to avoid hallucinations.

### Persistent Chroma Database

* Embeddings are generated once during ingestion.
* ChromaDB is stored locally and reused across sessions.

### Conversational Memory

* Maintains chat history across multiple turns.
* Supports follow-up questions such as:

```text
User: What is GIS?

User: What are its applications?
```

* Previous conversation is injected into the prompt to provide conversational context.
* Memory helps interpret references such as:

  * "it"
  * "they"
  * "that topic"

### LCEL Chains

Uses LangChain Expression Language (LCEL):

```text
PromptTemplate
      ↓
Chat Model
      ↓
Output Parser
```

### Retrieval Debug Mode

Optional debug mode for inspecting:

* Retrieved documents
* Metadata
* Retrieved chunk content

This helps determine whether issues originate from retrieval or generation.

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
Retrieved Context

Conversation History
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

### 5. Conversation history is prepared

```python
history = "\n\n".join(chat_history)
```

### 6. Prompt is sent to the LLM

```python
chain.invoke(...)
```

### 7. Response is generated

```text
GIS is a system used for storing, analyzing,
and visualizing geographic information.
```

### 8. Sources are displayed

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


ENTER YOUR QUESTION:

What are its applications?

- Urban planning
- Transportation management
- Environmental monitoring
- Resource management

Sources:
GIS_notes.pdf | Page 5
GIS_Assignment.pdf | Page 2
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
* Metadata management
* LCEL chains
* Retrieval debugging techniques
* Multi-turn conversation handling
* Conversation memory management
* Context vs Memory separation

---

## Key Concepts Learned

### Retrieval vs Memory

One of the most important concepts learned during this project was the distinction between retrieval and memory.

#### Retrieval

Responsible for:

* Fetching relevant information from documents
* Grounding answers in the knowledge base
* Providing source citations

#### Memory

Responsible for:

* Maintaining conversational flow
* Resolving references such as:

  * "it"
  * "they"
  * "that topic"
* Supporting follow-up questions

Memory improves conversation quality, but factual information should still come from retrieved documents rather than previous responses.

---

## Future Improvements

* Retrieval confidence scoring
* Score-threshold filtering
* Streamlit UI
* Hybrid Search (Keyword + Vector Search)
* Reranking
* Full LCEL retrieval chains
* Memory summarization for long conversations
* Evaluation metrics for retrieval quality

---

## Key Takeaway

A good RAG system is not only about generating answers.

It is equally about:

* Retrieving the correct information
* Grounding responses in source documents
* Maintaining conversational context
* Providing transparency through citations

This project helped bridge the gap between a basic PDF chatbot and a more realistic knowledge-grounded conversational AI system.
