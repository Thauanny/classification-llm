"""
FastAPI — aplicação principal.
Registra rotas, middleware CORS e endpoint de health check.
"""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..models.api_models import HealthResponse
from .dependencies import get_ollama_service
from .routes import classify, models
from ..services.ollama_service import OllamaService

app = FastAPI(
    title="Ollama Classification API",
    description=(
        "API para classificação de textos usando modelos LLM locais via Ollama. "
        "Suporta classificação de texto único e de datasets (CSV/Excel)."
    ),
    version="1.0.0",
    contact={"name": "Projeto Ollama LLM"},
    license_info={"name": "MIT"},
)

# CORS aberto para o Streamlit (e qualquer cliente local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router)
app.include_router(models.router)


@app.get(
    "/",
    tags=["Root"],
    summary="Bem-vindo",
)
def root() -> dict:
    return {
        "name": "Ollama Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Verifica se a API e o Ollama estão funcionando corretamente.",
)
def health_check(
    ollama: OllamaService = Depends(get_ollama_service),
) -> HealthResponse:
    connected = ollama.is_connected()
    available_models = ollama.list_models() if connected else []

    return HealthResponse(
        status="ok" if connected else "degraded",
        ollama_connected=connected,
        models_count=len(available_models),
        message=(
            f"Ollama conectado com {len(available_models)} modelo(s) disponível(is)."
            if connected
            else "Ollama não está acessível. Execute: ollama serve"
        ),
    )
