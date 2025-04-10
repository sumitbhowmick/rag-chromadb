# rag-chromadb

# 🧠 RAG Pipeline with ChromaDB, Ollama & LLaMA3.2

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:
- [ChromaDB](https://www.trychroma.com/) as a local vector database
- [Ollama](https://ollama.com/) to run the `llama3.2` LLM locally
- PDF and TXT file ingestion
- LangChain's `OllamaEmbeddings` to convert text into vector embeddings for semantic search

---

## 🚀 Features

- ✅ Read `.pdf` and `.txt` files from a specified directory
- ✅ Split large documents into context-aware chunks
- ✅ Embed and store documents in ChromaDB
- ✅ Query ChromaDB for relevant chunks using semantic similarity
- ✅ Pass retrieved context to LLaMA3.2 via Ollama for accurate and grounded answers

---

## 📦 Requirements

Install required dependencies:

```bash
pip install chromadb langchain langchain-community sentence-transformers pymupdf
```

Also ensure you have:
Ollama installed and running
The llama3.2 model pulled locally via:
```bash
ollama pull llama3:instruct
```
---

## ⚙️ Configuration

🔧 Ollama Model
Make sure llama3.2 is pulled:

```bash
ollama pull llama3.2
ollama run llama3.2
```
---

## 🧠 Embeddings
The project uses OllamaEmbeddings from LangChain to vectorize document chunks before storing them in ChromaDB.

---

## 🧪 Usage
1. Add PDF and TXT Files
Place your .pdf and .txt files in the documents/ folder.

2. Run the RAG Pipeline
```bash
python main.py
```
The script will:
Read and chunk all documents from /documents
Embed and store them in ChromaDB
Query the database and generate a response from LLaMA3.2

3. Customize the Query
Modify the query in main.py:
query = "What is Speedy Sid story?"
