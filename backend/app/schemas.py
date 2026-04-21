from enum import StrEnum

from pydantic import BaseModel


class Engine(StrEnum):
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"


class ChatRequest(BaseModel):
    message: str
    engine: Engine


class ChatResponse(BaseModel):
    reply: str
    engine: Engine
