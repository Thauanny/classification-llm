"""
Interface para Large Language Models.
Segue o princípio de Dependency Inversion (SOLID).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class ILLMService(ABC):
    """
    Interface abstrata para serviços de LLM.
    Permite a substituição de implementações sem afetar o código cliente (Open/Closed Principle).
    """

    @abstractmethod
    def pull_model(self, model_name: str) -> bool:
        """
        Baixa um modelo LLM.
        
        Args:
            model_name: Nome do modelo a ser baixado
            
        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário
        """
        pass

    @abstractmethod
    def is_model_available(self, model_name: str) -> bool:
        """
        Verifica se um modelo está disponível localmente.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            bool: True se o modelo está disponível, False caso contrário
        """
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, str]]:
        """
        Lista todos os modelos disponíveis.
        
        Returns:
            List[Dict]: Lista de modelos com suas informações
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Verifica se o serviço LLM está acessível.

        Returns:
            bool: True se o serviço está disponível
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> str:
        """
        Gera uma resposta usando o modelo.

        Args:
            prompt: Texto de entrada
            model_name: Nome do modelo
            temperature: Controla a aleatoriedade (0.0 a 2.0)
            max_tokens: Número máximo de tokens na resposta
            top_p: Nucleus sampling (0.0 a 1.0)
            top_k: Top-K sampling
            repeat_penalty: Penalidade por repetição de tokens

        Returns:
            str: Resposta gerada pelo modelo
        """
        pass


class ILogger(ABC):
    """
    Interface abstrata para logging.
    Permite diferentes implementações de log (arquivo, console, etc.).
    """

    @abstractmethod
    def info(self, message: str) -> None:
        """Registra uma mensagem informativa."""
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        """Registra um aviso."""
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """Registra um erro."""
        pass

    @abstractmethod
    def debug(self, message: str) -> None:
        """Registra uma mensagem de debug."""
        pass
