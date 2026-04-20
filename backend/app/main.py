import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ENV_PATH
from app.schemas import ChatRequest, ChatResponse, Engine
from app.services.langchain_rag.service import LangchainSrv

# Load credentials once at process start — callers should not trigger dotenv
# as a side effect of every query.
load_dotenv(ENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

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
