import asyncio
from src.agents.conversation_agent import conversation_agent_tool
from src.agents.policy_agent import policy_agent_tool
from src.database.memory_db import MemoryDB
from src.chains.agent_chains import create_aggregator_chain

aggregator_chain = create_aggregator_chain()
memory_db = MemoryDB()

async def run_agents_parallel(
        query,
        guest_type,
        loyalty,
        city,
        chat_history_tuples,
        chat_history_text
):
    policy_task = asyncio.to_thread(
        policy_agent_tool.invoke,
        {
            "input": query,
            "guest_type": guest_type,
            "loyalty": loyalty,
            "city": city,
            "chat_history": chat_history_tuples
        }
    )

    conversation_task = asyncio.to_thread(
        conversation_agent_tool.invoke,
        {
            "query": query,
            "conversation_memory": chat_history_text
        }
    )
    policy_result, conversation_result = await asyncio.gather(policy_task, conversation_task)

    return policy_result, conversation_result


async def agentic_rag_answer(
    query,
    guest_type,
    loyalty,
    city,
    session_id
    ):
        chat_history_tuples = memory_db.get_chat_history_tuples(session_id)
        chat_history_text = memory_db.get_chat_history_text(session_id)

        policy_output, conversation_output = await run_agents_parallel(
            query,
            guest_type,
            loyalty,
            city,
            chat_history_tuples,
            chat_history_text
        )

        final_answer = aggregator_chain.invoke({
            "policy_output": policy_output,
            "conversation_output": conversation_output,
            "question": query
        })

        memory_db.store_memory(session_id, "user", query)
        memory_db.store_memory(session_id, "assistant", final_answer['text'])

        return {
              "answer": final_answer['text'],
              "session_id": session_id,
              "policy_output": policy_output,
              "conversation_output": conversation_output
        }