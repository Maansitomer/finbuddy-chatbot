import os
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embedding_model
from utils.data_loader import load_all_datasets

DEFAULT_SAVE_PATH = "vectorstore/faiss_store"

def create_vector_store(internal_path, external_path):
    """Create a FAISS vector store from Document objects."""
    docs = load_all_datasets(internal_path, external_path)
    embeddings = get_embedding_model()
    return FAISS.from_documents(docs, embeddings)

def save_vector_store(store, save_path=DEFAULT_SAVE_PATH):
    """Save the FAISS vector store locally."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    store.save_local(save_path)

def load_vector_store(save_path=DEFAULT_SAVE_PATH):
    """Load a FAISS vector store from local storage, or create if missing."""
    embeddings = get_embedding_model()

    try:
        # Try loading existing index
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ Could not load vector store: {e}")
        print("🔄 Rebuilding FAISS index from datasets...")

        # Rebuild from datasets
        store = create_vector_store(internal_path="data/internal_docs", external_path="data/external_docs")
        save_vector_store(store, save_path)
        return store
