"""
Rotas de classificação via Groq.
Single Responsibility: apenas endpoints Groq.
"""

import io

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ...models.api_models import (
    GroqClassifyTextRequest,
    GroqClassifyTextResponse,
    ClassifyDatasetResponse,
    ClassifyResult,
)
from ...services.groq_service import GroqService
from ...services.classification_service import ClassificationService
from ...utils.logger import ConsoleLogger

router = APIRouter(prefix="/classify/groq", tags=["Classificação Groq"])

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _get_groq_service(api_key: str) -> GroqService:
    return GroqService(api_key=api_key, logger=ConsoleLogger(name="Groq"))


def _get_classification_service(groq: GroqService) -> ClassificationService:
    return ClassificationService(
        llm_service=groq,
        logger=ConsoleLogger(name="ClassificationGroq"),
    )


def _load_dataframe(file: UploadFile, content: bytes) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Formato não suportado. Use: {', '.join(_SUPPORTED_EXTENSIONS)}",
    )


@router.post(
    "/text",
    response_model=GroqClassifyTextResponse,
    summary="Classificar texto único via Groq",
    description=(
        "Classifica um texto usando um modelo Groq. "
        "A chave de API Groq deve ser enviada no corpo da requisição."
    ),
)
def groq_classify_text(request: GroqClassifyTextRequest) -> GroqClassifyTextResponse:
    groq = _get_groq_service(request.api_key)

    if not groq.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Não foi possível conectar à API Groq. Verifique a chave de API.",
        )

    service = _get_classification_service(groq)

    from ...models.api_models import OllamaParams
    params = OllamaParams(
        temperature=request.params.temperature,
        top_p=request.params.top_p,
        max_tokens=request.params.max_tokens,
    )

    try:
        result = service.classify_text_with_metadata(
            text=request.text,
            prompt_template=request.prompt_template,
            model_name=request.model_name,
            params=params,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return GroqClassifyTextResponse(
        text=request.text,
        classification=result["classification"],
        model_name=request.model_name,
        prompt_template=request.prompt_template,
        metadata=result.get("metadata"),
    )


@router.post(
    "/dataset",
    response_model=ClassifyDatasetResponse,
    summary="Classificar dataset via Groq",
    description="Recebe um arquivo CSV/Excel e classifica cada linha via Groq.",
)
async def groq_classify_dataset(
    file: UploadFile = File(...),
    column_name: str = Form(...),
    prompt_template: str = Form(...),
    model_name: str = Form("llama-3.3-70b-versatile"),
    api_key: str = Form(..., description="Chave de API Groq."),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    max_tokens: int = Form(0, description="0 = ilimitado"),
) -> ClassifyDatasetResponse:
    groq = _get_groq_service(api_key)

    if not groq.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Não foi possível conectar à API Groq. Verifique a chave de API.",
        )

    content = await file.read()
    df = _load_dataframe(file, content)

    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Coluna '{column_name}' não encontrada. Disponíveis: {list(df.columns)}",
        )

    from ...models.api_models import OllamaParams
    params = OllamaParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens if max_tokens > 0 else None,
    )

    service = _get_classification_service(groq)
    texts = df[column_name].dropna().astype(str).tolist()
    results = service.classify_texts(
        texts=texts,
        prompt_template=prompt_template,
        model_name=model_name,
        params=params,
    )

    return ClassifyDatasetResponse(
        results=[ClassifyResult(index=r.index, text=r.text, classification=r.classification) for r in results],
        total=len(results),
        model_name=model_name,
        errors=sum(1 for r in results if r.classification.startswith("ERRO:")),
    )
