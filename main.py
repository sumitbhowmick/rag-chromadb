# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import fitz  # PyMuPDF

# Define the LLM model to be used
llm_model = "llama3.2"  #llama3.2

# Configure ChromaDB
# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

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
    "ChromaDB is a vector database designed for AI applications."
]
doc_ids = ["doc1", "doc2", "doc3"]

# Documents only need to be added once or whenever an update is required. 
# This line of code is included for demonstration purposes:
add_documents_to_collection(documents, doc_ids)

# Add all PDFs from /documents
data_directory  = "documents"
add_files_from_directory(data_directory)

############---DEBUG ONLY START ---############
print("Total documents in collection:", collection.count())
sample = collection.query(query_texts=["Speedy Sid"], n_results=3)
print("Query results sample:", sample["documents"])


############---DEBUG ONLY END ---############

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
def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

# Example usage
# Define a query to test the RAG pipeline
query = "Who is Chikki in Speedy Sid?"  # Change the query as needed
response = rag_pipeline(query)
print("######## Response from LLM ########\n", response)