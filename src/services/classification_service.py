"""
Serviço de classificação de textos usando LLM.
Single Responsibility: apenas lógica de classificação.
Dependency Inversion: depende de ILLMService e ILogger abstratos.
"""

from typing import List

from ..interfaces.llm_interface import ILLMService, ILogger
from ..models.api_models import OllamaParams, ClassifyResult


class ClassificationService:
    """
    Serviço responsável por classificar textos usando um modelo LLM.

    Princípios SOLID:
    - Single Responsibility: apenas classificação de textos
    - Dependency Inversion: depende de abstrações (ILLMService, ILogger)
    - Open/Closed: extensível via herança sem modificar esta classe
    """

    def __init__(self, llm_service: ILLMService, logger: ILogger) -> None:
        """
        Args:
            llm_service: Serviço LLM injetado (Dependency Injection)
            logger: Logger injetado (Dependency Injection)
        """
        self._llm = llm_service
        self._logger = logger
        self._logger.info("ClassificationService inicializado.")

    def classify_text(
        self,
        text: str,
        prompt_template: str,
        model_name: str,
        params: OllamaParams,
    ) -> str:
        """
        Classifica um único texto usando o modelo LLM.

        Args:
            text: Texto a ser classificado
            prompt_template: Template do prompt (use {text} como placeholder)
            model_name: Nome do modelo Ollama
            params: Parâmetros de geração

        Returns:
            str: Classificação retornada pelo modelo

        Raises:
            ValueError: Se o modelo não está disponível
            RuntimeError: Se houver falha na comunicação com Ollama
        """
        full_prompt = self._build_prompt(text, prompt_template)
        self._logger.debug(f"Classificando texto ({len(text)} chars): {text[:60]}...")

        result = self._llm.generate(
            prompt=full_prompt,
            model_name=model_name,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
        )

        return result.strip()

    def classify_texts(
        self,
        texts: List[str],
        prompt_template: str,
        model_name: str,
        params: OllamaParams,
    ) -> List[ClassifyResult]:
        """
        Classifica uma lista de textos sequencialmente.

        Args:
            texts: Lista de textos a classificar
            prompt_template: Template do prompt
            model_name: Nome do modelo Ollama
            params: Parâmetros de geração

        Returns:
            List[ClassifyResult]: Resultados com índice, texto e classificação
        """
        results: List[ClassifyResult] = []
        total = len(texts)

        for i, text in enumerate(texts):
            self._logger.info(f"Classificando {i + 1}/{total}: {text[:50]}...")
            try:
                classification = self.classify_text(text, prompt_template, model_name, params)
                results.append(
                    ClassifyResult(index=i, text=text, classification=classification)
                )
            except (ValueError, RuntimeError) as exc:
                self._logger.error(f"Erro ao classificar texto {i}: {exc}")
                results.append(
                    ClassifyResult(index=i, text=text, classification=f"ERRO: {exc}")
                )

        return results

    def _build_prompt(self, text: str, prompt_template: str) -> str:
        """
        Constrói o prompt final substituindo o placeholder {text}.
        Se {text} não estiver no template, o texto é anexado ao final.

        Args:
            text: Texto a ser inserido no prompt
            prompt_template: Template fornecido pelo usuário

        Returns:
            str: Prompt final completo
        """
        if "{text}" in prompt_template:
            return prompt_template.format(text=text)
        return f"{prompt_template}\n\nTexto: {text}"
