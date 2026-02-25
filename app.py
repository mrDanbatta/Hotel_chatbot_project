from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback
from src.agents.aggregator_agent import agentic_rag_answer
from src.database.memory_db import MemoryDB

app = FastAPI(title="Intelligent Concierge System API")

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://54.221.168.236:8501", # streamlit frontend
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatRequest(BaseModel):
    query: str
    guest_type: str
    loyalty: str
    city: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    policy_output: Optional[dict] = None
    conversation_output: Optional[dict] = None
    success: bool = True

class ClearMemoryRequest(BaseModel):
    session_id: str


# initialize memory db
memory_db = MemoryDB()

@app.get("/")
async def root():
    return {"message": "Welcome to the Intelligent Concierge System API!"}  

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print(f"\n=== CHAT REQUEST ===")
        print(f"Query: {request.query}")
        print(f"Session ID: {request.session_id}")
        print(f"Guest Type: {request.guest_type}")
        print(f"Loyalty: {request.loyalty}")
        print(f"City: {request.city}")
        
        result = await agentic_rag_answer(
            query=request.query,
            guest_type=request.guest_type,
            loyalty=request.loyalty,
            city=request.city,
            session_id=request.session_id
        )
        
        print(f"Response: {result.get('answer', 'NO ANSWER')[:100]}")
        print("=== SUCCESS ===\n")
        
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
            success=True,
        )
    except Exception as e:
        print(f"\n=== ERROR IN /chat ENDPOINT ===")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("Full Traceback:")
        traceback.print_exc()
        print("===========================\n")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    try:
        memory_tuples = memory_db.get_chat_history_tuples(session_id)
        memory_text = memory_db.get_chat_history_text(session_id)

        return {
            "session_id": session_id,
            "memory_tuples": memory_tuples,
            "memory_text": memory_text,
            "success": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/clearmemory")
async def clear_memory(request: ClearMemoryRequest):
    try:
        memory_db.clear_memory(request.session_id)
        return {
            "message": f"Memory cleared for session",
            "session_id": request.session_id,
            "success": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)