# rag-chromadb

# 🧠 RAG Pipeline with ChromaDB, Ollama, LLaMA3.2 and Weaviate(optional)

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using:
- [ChromaDB](https://www.trychroma.com/) as a local vector database
- [Ollama](https://ollama.com/) to run the `llama3.2` LLM locally
- PDF and TXT file ingestion
- LangChain's `OllamaEmbeddings` to convert text into vector embeddings for semantic search
- [Weaviate](https://weaviate.io/) as a containerized local vector database with built-in reranking capability
---

## 🚀 Project Overview
The system is designed to:
- 📌 Ingest Documents: Read text from PDF and TXT files.
- 📌 Chunk Text: Split documents into manageable chunks for processing.
- 📌 Embed Text: Convert text chunks into vector embeddings.
- 📌 Store Embeddings: Save embeddings in a vector database (ChromaDB or Weaviate).
- 📌 Semantic Search: Retrieve relevant document chunks based on query similarity.
- 📌 Rerank Results: Optionally rerank search results using cross-encoders.
- 📌 Generate Responses: Use a language model to generate responses based on retrieved context.

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
Ollama installed and running.

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
```bash
query = "What is Speedy Sid story?"
```
4. Modify the Response type by tweaking context size. Abstract queries require higher context(2-5). Higher values will result in more broad based response, abstract responses, but may be incorrect at times. Lower will be specific, but it may miss information in case of complex queries.
```bash
context_size = 3
```

## 🧠 Using Weaviate with Docker
To use Weaviate as the vector database, follow these steps:

### Install Docker:

Ensure Docker is installed on your system. You can download it from the official Docker website.
Download Weaviate's Docker Compose File:
Use the Weaviate Docker Compose configurator to generate a docker-compose.yml file tailored to your needs. Access the configurator here.

### Start Weaviate:

Navigate to the directory containing your docker-compose.yml file and run:
```bash
docker compose up -d
```
This command starts Weaviate in detached mode.

### Install Weaviate Python Client:
```bash
pip install weaviate-client
```

## Document Ingestion
The system reads documents from a specified directory, supporting both PDF and TXT formats. It extracts text content and prepares it for processing.

## Semantic Search and Reranking
Upon receiving a query, the system performs a semantic search to retrieve relevant document chunks. If configured, it uses a cross-encoder model to rerank the search results for improved relevance.

## Querying the Language Model
The system integrates with a language model to generate responses based on the retrieved context. It constructs a prompt that includes the context and the user's query, then invokes the language model to produce an answer.

## Retrieval-Augmented Generation Pipeline
The complete RAG pipeline involves:
- ✅ Retrieving relevant document chunks using semantic search.
- ✅ Optionally reranking the retrieved chunks.
- ✅ Generating a response using the language model, informed by the retrieved context.

### Running the Example
With ChromaDB -
```bash
python main.py
```
With Weaviate -
```bash
python weave.py
```
