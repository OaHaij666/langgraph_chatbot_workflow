from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import json
import uvicorn
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Python:", sys.version, file=sys.stderr)
print("Starting import config...", file=sys.stderr)
from config import get_config
print("Config imported", file=sys.stderr)
from app.llm import get_llm_client
print("LLM client imported", file=sys.stderr)

app = FastAPI(title="AI Chat System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = None


@app.on_event("startup")
async def startup_event():
    global llm_client
    provider = get_config("llm.provider", "openai")
    if provider == "unsloth":
        print(f"Loading {provider} model...")
        llm_client = get_llm_client()
        if hasattr(llm_client, "_client") and hasattr(llm_client._client, "_load_model"):
            llm_client._client._load_model()
        print(f"Model loaded successfully")
    else:
        llm_client = get_llm_client()
        if provider == "ollama":
            try:
                import requests
                base_url = getattr(llm_client._client, "base_url", "http://localhost:11434")
                requests.get(f"{base_url}/api/tags", timeout=3)
            except Exception:
                pass


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    output_type: str = "text"
    persona: Optional[str] = None
    multimodal_inputs: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    message: str
    intent: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Chat System is running"}


def build_streaming_response(request: ChatRequest):
    async def generate():
        try:
            for chunk in llm_client.stream_chat(
                request.message,
                multimodal_inputs=request.multimodal_inputs,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if llm_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if request.stream:
            return build_streaming_response(request)
        message = llm_client.chat(request.message, multimodal_inputs=request.multimodal_inputs)
        return ChatResponse(message=message, intent="chat")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = get_config("server.port", 721)
    uvicorn.run(app, host="0.0.0.0", port=port)
