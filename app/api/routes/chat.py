from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.llm.llm_service import LLMService
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.vector_store import VectorStore
from app.services.retrieval.keyword_search import KeywordSearchEngine
from app.services.retrieval.hybrid_search import HybridRetriever
from app.services.retrieval.context_builder import RetrievalContextBuilder
from app.services.orchestration.chat_orchestrator import ChatOrchestrator

router = APIRouter()

# Singleton orchestrator — initialized once on first request
_orchestrator: ChatOrchestrator | None = None


def get_orchestrator() -> ChatOrchestrator:
    """Returns the singleton ChatOrchestrator, initializing on first call."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing ChatOrchestrator and sub-services...")
        llm = LLMService()
        embeddings = EmbeddingService()
        v_store = VectorStore()
        v_store.load_index()  # Graceful: returns False if not found, no crash
        k_engine = KeywordSearchEngine()
        k_engine.load()       # Graceful: returns False if not found, no crash

        retriever = HybridRetriever(v_store, k_engine, embeddings)
        context_builder = RetrievalContextBuilder()

        _orchestrator = ChatOrchestrator(llm, retriever, context_builder)
        logger.info("ChatOrchestrator ready.")
    return _orchestrator


@router.post("", response_model=ChatResponse)
async def chat_with_recommender(request: ChatRequest):
    """
    POST /chat — Main entry point for conversational assessment recommendations.

    Accepts the full stateless conversation history as a messages[] array:
      { "messages": [{"role": "user", "content": "..."}, ...] }

    Returns the agent's reply, structured recommendations, and end_of_conversation flag.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty")

    # Validate at least one user message exists
    user_msgs = [m for m in request.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found in messages array")

    orchestrator = get_orchestrator()

    try:
        response = await orchestrator.handle_chat(request)
        return response
    except Exception as e:
        logger.error(f"Error in chat orchestration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )
