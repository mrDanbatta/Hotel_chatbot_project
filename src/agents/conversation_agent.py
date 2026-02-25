from langchain.tools import tool
from src.chains.agent_chains import create_conversation_chain

# create chain instance
conversation_chain = create_conversation_chain()

@tool
def conversation_agent_tool(
    query: str,
    conversation_memory: str
    ):
    """
    Use this tool to analyse historical guest-assistant conversations.
    It identifies how similar questions were handled in the past, including tone, uncertainty handling, escalation patterns and 
    typical assistant responses. This tool does not interpret policy and should be used to inform response style  and guest experience only.
    """
    return conversation_chain.invoke({
        "query": query,
        "chat_history": conversation_memory
    })["result"]