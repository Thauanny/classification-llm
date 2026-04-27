"""
Injeção de dependências para a API FastAPI.
Dependency Inversion Principle: a API depende de abstrações, não de implementações.
"""

from functools import lru_cache

from ..services.ollama_service import OllamaService
from ..services.classification_service import ClassificationService
from ..utils.logger import ConsoleLogger
from ..interfaces.llm_interface import ILogger


@lru_cache(maxsize=1)
def get_logger() -> ILogger:
    """Retorna o logger singleton da aplicação."""
    return ConsoleLogger(name="API")


@lru_cache(maxsize=1)
def get_ollama_service() -> OllamaService:
    """Retorna o OllamaService singleton."""
    return OllamaService(logger=get_logger())


@lru_cache(maxsize=1)
def get_classification_service() -> ClassificationService:
    """Retorna o ClassificationService singleton."""
    return ClassificationService(
        llm_service=get_ollama_service(),
        logger=get_logger(),
    )
