import os
from dotenv import load_dotenv

load_dotenv(override=True)


def get_config():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "travel-assistant-documents"),
        "PINECONE_CONVERSATION_INDEX": os.getenv("PINECONE_CONVERSATION_INDEX", "travel-assistant-conversation"),
        "OPENAI_MODEL": "gpt-4o-mini",
        "DB_PATH": "conversation_memory.db"
    }