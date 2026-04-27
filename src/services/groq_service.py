"""
Serviço para interação com a API Groq.
Implementa ILLMService para ser intercambiável com OllamaService (LSP).
"""

from typing import Dict, List, Optional

import requests

from ..interfaces.llm_interface import ILLMService, ILogger

# Modelos disponíveis no free tier do Groq
GROQ_FREE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
]

_GROQ_API_BASE = "https://api.groq.com/openai/v1"


class GroqService(ILLMService):
    """
    Cliente para a API Groq (compatível com OpenAI).
    Usa o endpoint /chat/completions com a chave de API fornecida pelo usuário.
    """

    def __init__(self, api_key: str, logger: ILogger):
        self._api_key = api_key
        self._logger = logger
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def is_connected(self) -> bool:
        try:
            resp = requests.get(
                f"{_GROQ_API_BASE}/models",
                headers=self._headers,
                timeout=5,
            )
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── Métodos de gerenciamento de modelo (sem suporte real no Groq) ─────────

    def pull_model(self, model_name: str) -> bool:
        """Groq não requer download de modelos — sempre True."""
        return True

    def is_model_available(self, model_name: str) -> bool:
        return model_name in self.list_model_names()

    def list_models(self) -> List[Dict[str, str]]:
        """Lista os modelos disponíveis na conta Groq."""
        try:
            resp = requests.get(
                f"{_GROQ_API_BASE}/models",
                headers=self._headers,
                timeout=5,
            )
            if resp.status_code != 200:
                return [{"name": m, "size": "cloud"} for m in GROQ_FREE_MODELS]
            data = resp.json().get("data", [])
            return [{"name": m["id"], "size": "cloud"} for m in data]
        except requests.exceptions.RequestException:
            return [{"name": m, "size": "cloud"} for m in GROQ_FREE_MODELS]

    def list_model_names(self) -> List[str]:
        return [m["name"] for m in self.list_models()]

    # ── Geração de texto ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        top_k: int = 40,          # não suportado pelo Groq — ignorado
        repeat_penalty: float = 1.1,  # não suportado diretamente — ignorado
    ) -> str:
        """
        Gera uma resposta via Groq usando o endpoint de chat completions.

        Args:
            prompt: Prompt completo já com o texto embutido
            model_name: ID do modelo Groq
            temperature: Aleatoriedade (0–2)
            max_tokens: Limite de tokens na resposta
            top_p: Nucleus sampling
            top_k: Ignorado pelo Groq
            repeat_penalty: Ignorado pelo Groq

        Returns:
            str: Texto gerado pelo modelo
        """
        if not self._api_key:
            raise ValueError("Chave de API Groq não configurada.")

        payload: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        self._logger.debug(f"[Groq] Gerando com modelo {model_name}…")

        try:
            resp = requests.post(
                f"{_GROQ_API_BASE}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Falha na comunicação com Groq: {exc}") from exc

        if resp.status_code == 401:
            raise ValueError("Chave de API Groq inválida ou expirada.")
        if resp.status_code == 429:
            raise RuntimeError("Rate limit Groq atingido. Aguarde e tente novamente.")
        if resp.status_code != 200:
            raise RuntimeError(f"Groq retornou status {resp.status_code}: {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_with_metadata(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> Dict:
        """
        Igual a generate() mas retorna também os metadados da resposta Groq
        (usage, finish_reason, tempos, etc.).

        Returns:
            dict com chaves 'text' (str) e 'metadata' (dict).
        """
        if not self._api_key:
            raise ValueError("Chave de API Groq não configurada.")

        payload: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            resp = requests.post(
                f"{_GROQ_API_BASE}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Falha na comunicação com Groq: {exc}") from exc

        if resp.status_code == 401:
            raise ValueError("Chave de API Groq inválida ou expirada.")
        if resp.status_code == 429:
            raise RuntimeError("Rate limit Groq atingido. Aguarde e tente novamente.")
        if resp.status_code != 200:
            raise RuntimeError(f"Groq retornou status {resp.status_code}: {resp.text}")

        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        completion_time = usage.get("completion_time")

        metadata: Dict = {
            "completion_id": data.get("id"),
            "request_id": (data.get("x_groq") or {}).get("id"),
            "created": data.get("created"),
            "model_used": data.get("model"),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
            "queue_time_s": (
                round(usage["queue_time"], 4) if usage.get("queue_time") else None
            ),
            "prompt_time_s": (
                round(usage["prompt_time"], 3) if usage.get("prompt_time") else None
            ),
            "completion_time_s": (
                round(completion_time, 3) if completion_time else None
            ),
            "total_time_s": (
                round(usage["total_time"], 3) if usage.get("total_time") else None
            ),
        }

        if completion_tokens and completion_time and completion_time > 0:
            metadata["tokens_per_second"] = round(completion_tokens / completion_time, 1)

        return {"text": text, "metadata": metadata}
