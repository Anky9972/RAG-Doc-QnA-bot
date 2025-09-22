import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "PdfChunk")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
WEAVIATE_CHAT_MESSAGE_CLASS_NAME = "ChatMessage"
WEAVIATE_CONVERSATION_CLASS_NAME = "Conversation"
# For sentence-transformers/all-MiniLM-L6-v2, the dimension is 384
EMBEDDING_DIM = 384

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")