from langchain_openai import OpenAIEmbeddings
from src.utils.config import get_config

def get_embeddings():
    config = get_config()
    return OpenAIEmbeddings(openai_api_key = config["OPENAI_API_KEY"])