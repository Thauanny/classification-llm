"""
Modelos Pydantic para a API FastAPI.
Define contratos de request/response (Single Responsibility).
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class OllamaParams(BaseModel):
    """Parâmetros de geração do modelo Ollama."""

    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Controla a aleatoriedade. 0 = determinístico, 2 = muito aleatório.",
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling. Considera apenas tokens com probabilidade acumulada até Top-P.",
    )
    top_k: int = Field(
        40,
        ge=1,
        le=500,
        description="Considera apenas os K tokens mais prováveis em cada passo.",
    )
    max_tokens: Optional[int] = Field(
        None,
        gt=0,
        description="Número máximo de tokens na resposta. None = ilimitado.",
    )
    repeat_penalty: float = Field(
        1.1,
        ge=0.0,
        le=3.0,
        description="Penalidade por repetição de tokens. > 1.0 reduz repetições.",
    )


class ClassifyTextRequest(BaseModel):
    """Request para classificação de texto único."""

    text: str = Field(..., min_length=1, description="Texto a ser classificado.")
    prompt_template: str = Field(
        ...,
        min_length=1,
        description=(
            "Template do prompt de classificação. "
            "Use {text} para indicar onde o texto será inserido. "
            "Exemplo: 'Classifique como Positivo ou Negativo: {text}'"
        ),
    )
    model_name: str = Field(
        "llama3.2:3b-instruct-fp16",
        description="Nome do modelo Ollama a ser usado.",
    )
    params: OllamaParams = Field(
        default_factory=OllamaParams,
        description="Parâmetros de geração do modelo.",
    )


class ClassifyTextResponse(BaseModel):
    """Response da classificação de texto único."""

    text: str
    classification: str
    model_name: str
    prompt_template: str


class ClassifyResult(BaseModel):
    """Resultado de uma classificação individual (para datasets)."""

    index: int = Field(description="Índice da linha no dataset original.")
    text: str
    classification: str


class ClassifyDatasetResponse(BaseModel):
    """Response da classificação de dataset."""

    results: List[ClassifyResult]
    total: int
    model_name: str
    errors: int = Field(0, description="Número de classificações com erro.")


class ModelPullRequest(BaseModel):
    """Request para baixar um modelo."""

    model_name: str = Field(..., description="Nome do modelo Ollama a ser baixado.")


class ModelInfoResponse(BaseModel):
    """Informações de um modelo disponível."""

    name: str
    size: str
    modified_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Response do health check da API."""

    status: str
    ollama_connected: bool
    models_count: int
    message: str
