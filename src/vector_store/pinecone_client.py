from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.vector_store.embeddings import get_embeddings
from src.utils.config import get_config

def init_pinecone():
    config = get_config()
    pinecone_client = Pinecone(api_key=config["PINECONE_API_KEY"])
    return pinecone_client

def get_policy_vectorstore():
    config = get_config()
    embeddings = get_embeddings()
    pinecone = init_pinecone()

    #check if index exists, if not create it
    if config["PINECONE_INDEX_NAME"] not in [i.name for i in pinecone.list_indexes()]:
        pinecone.create_index(
            name=config["PINECONE_INDEX_NAME"], 
            dimension=1563,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    return PineconeVectorStore.from_existing_index(
        index_name=config["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )


def get_conversation_vectorstore():
    config = get_config()
    embeddings = get_embeddings()
    pinecone = init_pinecone()

    #check if index exists, if not create it
    if config["PINECONE_CONVERSATION_INDEX"] not in [i.name for i in pinecone.list_indexes()]:
        pinecone.create_index(
            name=config["PINECONE_CONVERSATION_INDEX"], 
            dimension=1563,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    return PineconeVectorStore.from_existing_index(
        index_name=config["PINECONE_CONVERSATION_INDEX"],
        embedding=embeddings
    )
