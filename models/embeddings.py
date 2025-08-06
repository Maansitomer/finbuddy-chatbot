from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from config/config.env
load_dotenv("config/config.env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY not found in config/config.env!")

def get_embedding_model():
    """
    Returns Gemini embedding model for vector store.
    Keep this same as the one used when creating FAISS index
    so that dimensions match.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
