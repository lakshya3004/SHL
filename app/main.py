import time
import json
import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.api.router import api_router
from app.api.routes import health, chat
from app.core.config import settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Auto-builds search indices on startup and logs shutdown events.
    """
    logger.info("Starting SHL Assessment Recommender...")
    try:
        await _ensure_indices_ready()
    except Exception as e:
        logger.error(f"Failed to build indices on startup: {e}")
        logger.warning("Service will start with degraded retrieval. Check GEMINI_API_KEY / LLM keys.")

    logger.info("Service startup complete.")
    yield
    logger.info("Shutting down SHL Recommender Service.")


def create_application() -> FastAPI:
    """
    Initialize the FastAPI application with configuration, middleware, and routers.
    """
    setup_logging()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="AI-powered SHL Assessment Recommender API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow all origins so the frontend can talk to the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.0f}ms)"
        )
        return response

    # ---- ROOT-LEVEL ROUTES (required by evaluator spec) ----
    # GET /health and POST /chat must be accessible at the root, not under /api/v1
    app.include_router(health.router, prefix="/health", tags=["system"])
    app.include_router(chat.router, prefix="/chat", tags=["recommendation"])

    # ---- Versioned API routes (for documentation) ----
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # ---- Serve frontend static files ----
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    return app


app = create_application()


async def _ensure_indices_ready():
    """
    Checks if retrieval indices exist. If not, builds them from the catalog seed JSON.
    This provides a self-healing startup so the service is always ready.
    """
    faiss_path = os.path.join(settings.VECTOR_STORE_PATH, "faiss.index")
    tfidf_path = os.path.join(settings.VECTOR_STORE_PATH, "tfidf.pkl")
    seed_path = settings.CATALOG_SEED_PATH

    if os.path.exists(faiss_path) and os.path.exists(tfidf_path):
        logger.info("Retrieval indices found on disk. Skipping rebuild.")
        return

    if not os.path.exists(seed_path):
        logger.warning(f"Catalog seed not found at {seed_path}. Cannot build indices.")
        return

    logger.info("Retrieval indices not found. Building from catalog seed...")

    # Lazy imports to keep startup fast if indices already exist
    from app.services.retrieval.embeddings import EmbeddingService
    from app.services.retrieval.vector_store import VectorStore
    from app.services.retrieval.keyword_search import KeywordSearchEngine

    with open(seed_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Build retrieval documents from catalog
    texts = []
    metadatas = []
    for item in catalog:
        # Rich text for embedding: name + type + description + keywords
        text = " ".join(filter(None, [
            item.get("name", ""),
            item.get("test_type", ""),
            item.get("description", ""),
            " ".join(item.get("keywords", []))
        ]))
        texts.append(text)

        meta = {
            "id": item.get("id", item.get("name", "").lower().replace(" ", "-")),
            "name": item.get("name", ""),
            "url": item.get("url", ""),
            "test_type": item.get("test_type", "General Assessment"),
            "description": item.get("description", ""),
            "duration": item.get("duration"),
            "remote_support": item.get("remote_support", False),
            "adaptive": item.get("adaptive", False),
            "keywords": item.get("keywords", []),
        }
        metadatas.append(meta)

    if not texts:
        logger.error("Catalog seed is empty. Cannot build indices.")
        return

    # Build FAISS index
    logger.info(f"Embedding {len(texts)} assessments...")
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_documents(texts)

    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
    vector_store = VectorStore(
        index_path=faiss_path,
        metadata_path=os.path.join(settings.VECTOR_STORE_PATH, "metadata.json")
    )
    vector_store.build_index(embeddings, metadatas)
    vector_store.save_index()

    # Build TF-IDF index
    keyword_engine = KeywordSearchEngine(model_path=tfidf_path)
    keyword_engine.fit(texts, metadatas)
    keyword_engine.save()

    logger.info(f"Indices built successfully. {len(texts)} assessments indexed.")
