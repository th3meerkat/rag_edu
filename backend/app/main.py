from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.langchain_rag.service import LangchainSrv
from app.schemas import ChatRequest, ChatResponse, Engine

app = FastAPI(title="rag-chat backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint: routes to the matching RagService based on the engine."""
    if req.engine == Engine.LANGCHAIN:
        reply = LangchainSrv().query(req.message)
    else:
        # reply = LlamaindexSrv().query(req.message)
        reply = "LLM response"
    return ChatResponse(reply=reply)
