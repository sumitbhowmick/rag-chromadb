# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import CrossEncoder
import chromadb
import os
import fitz  # PyMuPDF

# Define the LLM model to be used
llm_model = "llama3.2"  #llama3.2

#embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Config: Toggle to force re-ingestion even if DB exists
REINGEST_DATA = False
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")

# Detect if the chroma db path already has data
is_existing_db = os.path.isdir(CHROMA_DB_DIR) and bool(os.listdir(CHROMA_DB_DIR))
print(f"ChromaDB persistence found: {is_existing_db}")

# Configure ChromaDB
# Initialize the ChromaDB client with persistent storage in the current directory
#chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)


# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Initialize the embedding function with Ollama embeddings
#embedding = ChromaDBEmbeddingFunction(embedding_model)

#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)

def read_pdf_text(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()

def read_txt_text(txt_path):
    """Read text content from a .txt file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ""


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    if is_existing_db and not REINGEST_DATA:
        print("Skipping ingestion — existing ChromaDB detected.")

    collection.add(
        documents=documents,
        ids=ids
    )
    


def add_files_from_directory(directory_path, prefix="doc"):
    """
    Reads all PDF and TXT files in a directory and adds them to the ChromaDB collection.
    
    Args:
        directory_path (str): Path to the directory containing PDF and TXT files.
        prefix (str): Prefix to use for document IDs.
    """

    if is_existing_db and not REINGEST_DATA:
        print("Skipping document ingestion — existing ChromaDB detected.")
        return

    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    supported_extensions = [".pdf", ".txt"]
    texts = []
    ids = []

    file_list = sorted([
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_extensions
    ])

    for idx, filename in enumerate(file_list):
        full_path = os.path.join(directory_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = read_pdf_text(full_path)
        elif ext == ".txt":
            text = read_txt_text(full_path)
        else:
            continue  # skip unsupported files

        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                ids.append(f"{prefix}_{idx + 1}_{i}")

    if texts:
        add_documents_to_collection(texts, ids)
    else:
        print("No valid PDF or TXT documents found to add to the collection.")

# Example: Add sample documents to the collection
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "ChromaDB is a vector database designed for AI applications.",
    "Television is called an Idiot Box because it is a box that hardly makes you learn to think.",
    "Grapes are a type of fruit that grow in clusters on vines.",
]
doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

# Documents only need to be added once or whenever an update is required. 
# This line of code is included for demonstration purposes:
add_documents_to_collection(documents, doc_ids)

# Add all PDFs from /documents
data_directory  = "documents"
add_files_from_directory(data_directory)


# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]


def query_rerank(collection, query, n=1):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # small and fast
    print("Total documents in collection:", collection.count())

    sample = collection.query(query_texts=[query], n_results=20)
    documents = sample["documents"][0]

    #sample = collection.query(query_texts=[query], n_results=5)
    #for i, doc in enumerate(documents):
    #    print(f"\nRaw Query Result {i + 1}:", doc)

    pairs = [[query, doc] for doc in documents]

    scores = model.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    
    context = "\n\n".join(sorted_docs[:n])  # Use top 3 reranked chunks
    #print(f"\nReranked Sorted Query Result:", context)
    return context

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text, n=1):
    context = query_rerank(collection, query_text, n)
    context = context if context else "No relevant documents found."
    prompt = f"Context: {context}\n\n Instructions: Answer should be given based on the information available in the context. If information is not available in the context, then mention it could not be found in the context and general knowledge is used.     \n\n Question: {query_text}\nAnswer:"
    print("\n######## Augmented Prompt ########")
    print(prompt)
    return query_ollama(prompt)

# Example usage
# Define a query to test the RAG pipeline
context_size = 1 # Abstract queries require bigger context(2-5), Higher values will result in more broad based response, at cost of halluciantion. Lower will be specific, but chance of missing information on complex queries.
query = "What is good gene hypothesis?"  # Change the query as needed
response = rag_pipeline(query,context_size)
print("######## Response from LLM ########\n", response)




