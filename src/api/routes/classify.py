"""
Rotas de classificação de textos.
Single Responsibility: apenas endpoints de classificação.
"""

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ...models.api_models import (
    ClassifyDatasetResponse,
    ClassifyResult,
    ClassifyTextRequest,
    ClassifyTextResponse,
    OllamaParams,
)
from ...services.classification_service import ClassificationService
from ...services.ollama_service import OllamaService
from ..dependencies import get_classification_service, get_ollama_service

router = APIRouter(prefix="/classify", tags=["Classificação"])

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _load_dataframe(file: UploadFile, content: bytes) -> pd.DataFrame:
    """Carrega DataFrame a partir do arquivo enviado."""
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Formato não suportado. Use: {', '.join(_SUPPORTED_EXTENSIONS)}",
    )


def _validate_model(model_name: str, ollama: OllamaService) -> None:
    """Valida se o modelo está disponível, levantando HTTPException se não estiver."""
    if not ollama.is_model_available(model_name):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Modelo '{model_name}' não está disponível localmente. "
                "Use POST /models/pull para baixá-lo."
            ),
        )


@router.post(
    "/text",
    response_model=ClassifyTextResponse,
    summary="Classificar texto único",
    description=(
        "Classifica um texto com base no prompt fornecido. "
        "Use {text} no prompt_template para indicar onde o texto será inserido."
    ),
)
def classify_text(
    request: ClassifyTextRequest,
    service: ClassificationService = Depends(get_classification_service),
    ollama: OllamaService = Depends(get_ollama_service),
) -> ClassifyTextResponse:
    _validate_model(request.model_name, ollama)

    try:
        classification = service.classify_text(
            text=request.text,
            prompt_template=request.prompt_template,
            model_name=request.model_name,
            params=request.params,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ClassifyTextResponse(
        text=request.text,
        classification=classification,
        model_name=request.model_name,
        prompt_template=request.prompt_template,
    )


@router.post(
    "/dataset",
    response_model=ClassifyDatasetResponse,
    summary="Classificar dataset (CSV / Excel)",
    description=(
        "Recebe um arquivo CSV ou Excel, seleciona a coluna de texto e classifica "
        "cada linha usando o modelo e prompt especificados. "
        "Todos os parâmetros Ollama podem ser ajustados via Form fields."
    ),
)
async def classify_dataset(
    file: UploadFile = File(..., description="Arquivo CSV ou Excel com os dados."),
    column_name: str = Form(..., description="Nome da coluna que contém os textos."),
    prompt_template: str = Form(
        ...,
        description="Template do prompt. Use {text} para o texto da linha.",
    ),
    model_name: str = Form("llama3.2:3b-instruct-fp16"),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    max_tokens: int = Form(0, description="0 = ilimitado"),
    repeat_penalty: float = Form(1.1),
    service: ClassificationService = Depends(get_classification_service),
    ollama: OllamaService = Depends(get_ollama_service),
) -> ClassifyDatasetResponse:
    _validate_model(model_name, ollama)

    content = await file.read()

    try:
        df = _load_dataframe(file, content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao ler o arquivo: {exc}",
        ) from exc

    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Coluna '{column_name}' não encontrada. "
                f"Colunas disponíveis: {list(df.columns)}"
            ),
        )

    texts = df[column_name].dropna().astype(str).tolist()

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"A coluna '{column_name}' não contém dados válidos.",
        )

    params = OllamaParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens if max_tokens > 0 else None,
        repeat_penalty=repeat_penalty,
    )

    results: list[ClassifyResult] = service.classify_texts(
        texts=texts,
        prompt_template=prompt_template,
        model_name=model_name,
        params=params,
    )

    errors = sum(1 for r in results if r.classification.startswith("ERRO:"))

    return ClassifyDatasetResponse(
        results=results,
        total=len(results),
        model_name=model_name,
        errors=errors,
    )
