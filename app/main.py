from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="AI Chat System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    output_type: str = "text"
    persona: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    intent: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Chat System is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/chat/stream")
async def chat_stream():
    raise HTTPException(status_code=501, detail="Not implemented yet")
