from langchain_classic.chains import RetrievalQA
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from src.vector_store.pinecone_client import get_policy_vectorstore, get_conversation_vectorstore
from src.utils.config import get_config



def init_llm():
    config = get_config()
    llm = OpenAI(model=config["OPENAI_MODEL"], openai_api_key=config["OPENAI_API_KEY"], temperature=0.0)
    return llm

def create_policy_chain():
    policy_prompt = PromptTemplate(
        input_variables=[
            "context",
            "input",
            "guest_type",
            "loyalty",
            "city"
        ],
        template="""
    You are an intelligent travel assistant for a hotel concierge.

    ROLE:
    - You interpret policy documents strictly as written, without making assumptions or inferences.
    - You provide concise, accurate answers based solely on the provided policy context.
    - If the policy does not explicitly address the question, you respond with "The policy does not specify."
    - You do not provide any information that is not directly supported by the policy text.
    - You do not make assumptions about the user's intent or needs beyond what is explicitly stated in the question and policy.

    CONTEXT:
    Guest_Type: {guest_type}
    Loyalty Tier: {loyalty}
    City: {city}

    POLICY DOCUMENTS:
    {context}

    QUESTION:
    {input}

    OUTPUT FORMAT (JSON ONLY):
    {{
        "policy_facts":[string],
        "limitations":[string],
        "applicability":[string],
        "confidence": number (0-1)
    }}
    """
    )
    llm = init_llm()
    return create_retrieval_chain(
        get_policy_vectorstore().as_retriever(search_kwargs={"k":3}),
        create_stuff_documents_chain(llm, prompt=policy_prompt)
    )

def create_conversation_chain():
    conversation_prompt = PromptTemplate(
    input_variables=[
        "context",
        "question"
    ],
    template="""
    You are an agent  that analyses historical conversations, read through how the agent handled similar questions in the past, and answer your 
    USER QUESTION based on the historical conversations. What you need to do is to check your historical conversations and read through the conversations
    and answer based on that question.

    ROLE
    - Analyse how similar guest questions were handled historically
    - Focus on tone, escalated patterns, uncertainty handling
    - Also get how the questions were resolved by analysing the conversation relating to the question

    HISTORICAL CONVERSATIONS:
    {context}

    USER QUESTION:
    {question}

    OUTPUT FORMAT (JSON ONLY):
    {{
        "observed_patterns": string,
        "response_style": string,
        "conversation": string,
        "confidence": number
    }}
    """
    )

    llm = init_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever=get_conversation_vectorstore().as_retriever(search_kwargs={"k":3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": conversation_prompt}
    )



def create_aggregator_chain():
    aggregator_prompt = PromptTemplate(
    input_variables=[
        "policy_output",
        "conversation_output",
        "question"
    ],
    template="""
    You are the FINAL RESPONSE AGENT for Atlas Horizon.
    Your role is to generate the possible customer-facing answer by combining policy rules  with historicl agent-guest conversations.
    Ensure that your final answer  is in english.
    CORE PRINCIPLES:

    Policy output defines  what is allowed  and disallowed. Never violate it.

    Conversation output provides real-world context, explanations and phrasing that may not be  explicitly atated  in the policy.

    If the conversation output containes additional  details not present in the policy, you may use  them only  if they do not contradict the policy and are relevant to the question.

    When policy  is unclear or confidence is below 0.7, adopt careful conditional language in your answer.input_types

    If conversation and policy  differ, policy always overrides facts, but conversations can still guide tone and explanation depth.

    INPUTS:

    Policy Agent Output (auhoritative rules):
    {policy_output}

    Conversation Agent Output (how similar questions were answered in real customer-care interaction):
    {conversation_output}

    User Question:
    {question}

    TASK:

    Synthesize a final response that:

    Is actually compliant with policy

    Reflects how agents have historically explained or handled this situation

    Sounds natural, human, and helpful

    Includes clarifications or caveats when needed

    FINAL ANSWER REQUIREMENTS:

    Length: 2-4 sentences

    Tone: clear, calm, customer-friendly

    Do not mention internal agents, policies, or confidence scores

    If Uncertainty exsists, gently suggest confirmaion

    FINAL ANSWER:
    """
    )

    llm = init_llm()

    return LLMChain(
        llm=llm,
        prompt=aggregator_prompt
    )

