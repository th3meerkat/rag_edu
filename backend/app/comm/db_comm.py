
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
COLLECTION_NAME = "langchain_rag"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

EMBEDDING_MODEL = "text-embedding-3-small"



def get_vectorstore() -> Chroma:
    """Instantiate the Langchain-Chroma vectorstore pointing at the service's collection."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata=COLLECTION_METADATA,
    )

def get_collection_count():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_collection(COLLECTION_NAME).count()