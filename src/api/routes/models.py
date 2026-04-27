"""
Rotas de modelos Ollama.
Single Responsibility: apenas gerenciamento de modelos.
"""

import json
from typing import Generator, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...models.api_models import ModelInfoResponse, ModelPullRequest
from ...services.ollama_service import OllamaService
from ..dependencies import get_ollama_service

router = APIRouter(prefix="/models", tags=["Modelos"])


@router.get(
    "/",
    response_model=List[ModelInfoResponse],
    summary="Listar modelos disponíveis",
    description="Retorna todos os modelos Ollama instalados localmente.",
)
def list_models(
    ollama: OllamaService = Depends(get_ollama_service),
) -> List[ModelInfoResponse]:
    available = ollama.list_models()
    return [ModelInfoResponse(**m) for m in available]


@router.post(
    "/pull",
    summary="Baixar modelo (sem progresso)",
    description="Aguarda o download completo e retorna sucesso/erro.",
)
def pull_model(
    request: ModelPullRequest,
    ollama: OllamaService = Depends(get_ollama_service),
) -> dict:
    if not ollama.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama não está acessível. Execute: ollama serve",
        )

    success = ollama.pull_model(request.model_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao baixar o modelo '{request.model_name}'.",
        )

    return {"message": f"Modelo '{request.model_name}' baixado com sucesso!"}


@router.post(
    "/pull/stream",
    summary="Baixar modelo com progresso (SSE)",
    description=(
        "Inicia o download e envia eventos de progresso via Server-Sent Events (SSE). "
        "Cada linha retornada é um JSON com: status, percent, completed, total, message."
    ),
)
def pull_model_stream(
    request: ModelPullRequest,
    ollama: OllamaService = Depends(get_ollama_service),
) -> StreamingResponse:
    if not ollama.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama não está acessível. Execute: ollama serve",
        )

    def _event_generator() -> Generator[str, None, None]:
        for progress in ollama.pull_model_stream(request.model_name):
            yield json.dumps(progress) + "\n"

    return StreamingResponse(
        _event_generator(),
        media_type="application/x-ndjson",
    )


@router.delete(
    "/{model_name:path}",
    summary="Remover modelo",
    description="Remove um modelo instalado localmente do Ollama.",
)
def delete_model(
    model_name: str,
    ollama: OllamaService = Depends(get_ollama_service),
) -> dict:
    if not ollama.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama não está acessível. Execute: ollama serve",
        )

    if not ollama.is_model_available(model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo '{model_name}' não encontrado localmente.",
        )

    success = ollama.delete_model(model_name)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao remover o modelo '{model_name}'.",
        )

    return {"message": f"Modelo '{model_name}' removido com sucesso!"}


@router.get(
    "/check/{model_name:path}",
    summary="Verificar disponibilidade de modelo",
    description="Verifica se um modelo específico está disponível localmente.",
)
def check_model(
    model_name: str,
    ollama: OllamaService = Depends(get_ollama_service),
) -> dict:
    available = ollama.is_model_available(model_name)
    return {
        "model_name": model_name,
        "available": available,
        "message": "Disponível" if available else "Não encontrado localmente.",
    }
