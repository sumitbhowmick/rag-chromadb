# Imports
import os
import fitz  # PyMuPDF
import uuid
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.connect import ConnectionParams


# Configuration
llm_model = "llama3.2"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)
ollama_embedding = OllamaEmbeddings(model=llm_model, base_url="http://localhost:11434")

# Weaviate client setup
#WEAVIATE_URL = "http://localhost:8080"
#client = WeaviateClient.connect_to_local(url=WEAVIATE_URL)
#client = weaviate.connect_to_local()
client = weaviate.connect_to_local(
        host="127.0.0.1",  # Use a string to specify the host
        port=8005,  # Default HTTP port 8080
        grpc_port=50051  # Default gRPC port
    )
print("Client Ready:",client.is_ready())


# Schema setup
CLASS_NAME = "RAGDoc"
REINGEST_DATA = False

existing_classes = [cls for cls in client.collections.list_all()]
if CLASS_NAME not in existing_classes:
    client.collections.create(
            CLASS_NAME,            
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT)
            ]
        
    )


# Text reading
def read_pdf_text(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()

def read_txt_text(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ""

# Chunking
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Add documents to Weaviate
def add_documents_to_weaviate(documents, reingest=False):
    collection = client.collections.get(CLASS_NAME)

    for doc in documents:
        vector = embedding_model.encode(doc).tolist()
        uuid_str = str(uuid.UUID(hashlib.md5(doc.encode("utf-8")).hexdigest()))

        try:
            # Check if object exists
            exists = collection.data.exists(uuid_str)
            if exists and not reingest:
                #print(f"Skipping existing document with UUID: {uuid_str}")
                continue

            # Delete old one if reingest
            if exists and reingest:
                collection.data.delete_by_id(uuid_str)

            # Insert new one
            collection.data.insert(
                uuid=uuid_str,
                properties={"text": doc},
                vector=vector
            )
        except Exception as e:
            print(f"Error inserting doc: {e}")



def add_files_from_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    supported_extensions = [".pdf", ".txt"]
    texts = []

    file_list = sorted([
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_extensions
    ])

    for filename in file_list:
        full_path = os.path.join(directory_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = read_pdf_text(full_path)
        elif ext == ".txt":
            text = read_txt_text(full_path)
        else:
            continue

        if text:
            chunks = chunk_text(text)
            texts.extend(chunks)

    if texts:
        add_documents_to_weaviate(texts)
    else:
        print("No valid documents found.")

# Semantic search and rerank
def query_rerank(query, top_k=3):
    vector = embedding_model.encode(query).tolist()
    collection = client.collections.get(CLASS_NAME)

    response = collection.query.near_vector(
        near_vector=vector,
        limit=10,
        return_properties=["text"]
    )

    results = [obj.properties["text"] for obj in response.objects]
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = reranker.predict([[query, doc] for doc in results])
    ranked = [doc for _, doc in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
    return "\n\n".join(ranked[:top_k])

# def query_rerank(query, top_k=3):
#     vector = embedding_model.encode(query).tolist()

#     response = client.collections.get(CLASS_NAME).query.near_vector(
#         near_vector={"vector": vector},
#         limit=10,
#         return_metadata=["rerank_score"],
#         rerank=True  # 🔥 Use Weaviate's semantic reranker
#     )

#     results = [obj.properties["text"] for obj in response.objects]
#     return "\n\n".join(results[:top_k])


# Query LLM with Ollama
def query_llm(prompt):
    llm = OllamaLLM(model=llm_model, base_url="http://localhost:11434")
    return llm.invoke(prompt)

# RAG pipeline
def rag_pipeline(query):
    context = query_rerank(query, top_k=2)
    augmented_prompt = f"Context: {context}\n\nInstructions: Answer based on the context. If not found, state that and use general knowledge.\n\nQuestion: {query}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)
    return query_llm(augmented_prompt)


try:
    # Ingest and query example
    docs_dir = os.path.join(os.getcwd(), "documents")
    add_files_from_directory(docs_dir)
    query = "What did Sid do on Day 2?"
    response = rag_pipeline(query)
    print("######## Response from LLM ########\n", response)
# Clean up Weaviate client
finally:
    client.close()

