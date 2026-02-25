from langchain.tools import tool
from src.chains.agent_chains import create_policy_chain

# create chain instance
policy_chain = create_policy_chain()

@tool
def policy_agent_tool(input: str, guest_type: str, loyalty: str, city: str, chat_history: list) -> dict:
    """
    Use this tool to interpret official hotel policies.
    It retrieves and sumarises authoritative policy documents relevant to the guest question, city, loyalty tier and guest type.
    The output reflects strict policy rules and limitations.
    """
    return policy_chain.invoke({
        "input": input,
        "chat_history": chat_history,
        "guest_type": guest_type,
        "loyalty": loyalty,
        "city": city
    })["answer"]