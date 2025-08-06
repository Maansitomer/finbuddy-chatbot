import os
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embedding_model
from utils.data_loader import load_all_datasets  # Your dataset loader

def load_vector_store(embeddings=None):
    """
    Loads the FAISS vector store if available, otherwise builds it from datasets.
    Works both locally and on Streamlit Cloud.
    
    Args:
        embeddings: Optional embeddings model. If not provided, will be created.
    """
    try:
        # If embeddings not passed, create them
        if embeddings is None:
            embeddings = get_embedding_model()

        # Path to FAISS store (relative to this file)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.normpath(os.path.join(base_dir, "../vectorstore/faiss_store"))

        # Check if FAISS index files exist
        index_file = os.path.join(faiss_path, "index.faiss")
        pkl_file = os.path.join(faiss_path, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(pkl_file):
            print(f"✅ Loading FAISS store from: {faiss_path}")
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        # If not found, rebuild from datasets
        print("⚠️ FAISS store not found — rebuilding from datasets...")
        texts = load_all_datasets()
        db = FAISS.from_texts(texts, embeddings)

        # Ensure folder exists
        os.makedirs(faiss_path, exist_ok=True)

        # Save the FAISS index for next time
        db.save_local(faiss_path)
        print(f"✅ FAISS store created and saved to: {faiss_path}")

        return db

    except Exception as e:
        print(f"❌ Could not load or build vector store: {e}")
        return None
