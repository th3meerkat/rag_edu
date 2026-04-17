from app.langchain_rag.service import LangchainSrv
from app.schemas import ChatRequest, Engine


def query(req: ChatRequest) -> str:
    if req.engine == Engine.LANGCHAIN:
        return LangchainSrv().query(req.message)
    else:
        # return LlamaindexSrv.query(req.message)
        return "LLM response"