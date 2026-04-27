"""
Serviço para interação com Ollama.
Implementa a interface ILLMService seguindo SOLID principles.
"""

import json
from typing import Dict, Generator, List, Optional

import requests

from ..interfaces.llm_interface import ILLMService, ILogger


class OllamaService(ILLMService):
    """
    Implementação do serviço Ollama.
    
    Princípios SOLID aplicados:
    - Single Responsibility: Apenas interação com Ollama
    - Open/Closed: Implementa ILLMService, extensível via herança
    - Liskov Substitution: Pode substituir ILLMService sem quebrar código
    - Dependency Inversion: Depende de ILogger abstrato
    """

    def __init__(
        self,
        logger: ILogger,
        base_url: str = "http://localhost:11434"
    ):
        """
        Inicializa o serviço Ollama.
        
        Args:
            logger: Logger para registrar eventos (Dependency Injection)
            base_url: URL base da API Ollama
        """
        self._logger = logger
        self._base_url = base_url
        self._api_url = f"{base_url}/api"
        self._logger.info(f"OllamaService inicializado com URL: {base_url}")

    def is_connected(self) -> bool:
        """
        Verifica se o Ollama está acessível.

        Returns:
            bool: True se conectado, False caso contrário
        """
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _check_connection(self) -> bool:
        """Alias privado para compatibilidade interna."""
        return self.is_connected()

    def pull_model(self, model_name: str) -> bool:
        """
        Baixa um modelo do Ollama.
        
        Args:
            model_name: Nome do modelo (ex: 'llama3.2:3b-instruct-fp16')
            
        Returns:
            bool: True se o download foi bem-sucedido
        """
        if not self._check_connection():
            self._logger.error("Ollama não está rodando. Execute: ollama serve")
            return False

        self._logger.info(f"Iniciando download do modelo: {model_name}")
        
        try:
            url = f"{self._api_url}/pull"
            payload = {"name": model_name}
            
            # Streaming response para mostrar progresso
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=None  # Sem timeout para downloads longos
            )
            
            if response.status_code != 200:
                self._logger.error(f"Erro ao baixar modelo: {response.text}")
                return False

            # Processa o streaming de progresso
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get('status', '')
                        
                        if 'pulling' in status.lower():
                            # Mostra progresso de download
                            progress = data.get('completed', 0)
                            total = data.get('total', 0)
                            if total > 0:
                                percentage = (progress / total) * 100
                                self._logger.info(
                                    f"Baixando: {percentage:.1f}% "
                                    f"({progress}/{total} bytes)"
                                )
                        elif status:
                            self._logger.info(f"Status: {status}")
                            
                        # Verifica se completou
                        if data.get('status') == 'success':
                            self._logger.info(
                                f"Modelo {model_name} baixado com sucesso!"
                            )
                            return True
                            
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Erro ao fazer requisição: {e}")
            return False

    def pull_model_stream(self, model_name: str) -> Generator[dict, None, None]:
        """
        Baixa um modelo e fornece progresso via generator.

        Cada item yielded é um dict com pelo menos:
          - status (str)
          - percent (float, 0–100)  — 0 quando não há total disponível
          - completed (int)
          - total (int)

        Args:
            model_name: Nome do modelo a baixar

        Yields:
            dict: Estado atual do download
        """
        if not self._check_connection():
            yield {"status": "error", "percent": 0, "completed": 0, "total": 0,
                   "message": "Ollama não está acessível. Execute: ollama serve"}
            return

        self._logger.info(f"Iniciando download streaming: {model_name}")
        try:
            response = requests.post(
                f"{self._api_url}/pull",
                json={"name": model_name},
                stream=True,
                timeout=None,
            )
            if response.status_code != 200:
                yield {"status": "error", "percent": 0, "completed": 0, "total": 0,
                       "message": f"HTTP {response.status_code}"}
                return

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                completed = data.get("completed", 0)
                total = data.get("total", 0)
                percent = (completed / total * 100) if total else 0
                yield {
                    "status": data.get("status", ""),
                    "percent": round(percent, 1),
                    "completed": completed,
                    "total": total,
                    "message": data.get("status", ""),
                }

                if data.get("status") == "success":
                    self._logger.info(f"Modelo {model_name} baixado com sucesso!")
                    return

        except requests.exceptions.RequestException as exc:
            self._logger.error(f"Erro no streaming de pull: {exc}")
            yield {"status": "error", "percent": 0, "completed": 0, "total": 0,
                   "message": str(exc)}

    def delete_model(self, model_name: str) -> bool:
        """
        Remove um modelo instalado localmente.

        Args:
            model_name: Nome do modelo a remover

        Returns:
            bool: True se removido com sucesso
        """
        if not self._check_connection():
            self._logger.error("Ollama não está acessível.")
            return False

        self._logger.info(f"Removendo modelo: {model_name}")
        try:
            response = requests.delete(
                f"{self._api_url}/delete",
                json={"name": model_name},
                timeout=30,
            )
            if response.status_code in (200, 204):
                self._logger.info(f"Modelo {model_name} removido com sucesso.")
                return True

            self._logger.error(f"Erro ao remover modelo: {response.text}")
            return False

        except requests.exceptions.RequestException as exc:
            self._logger.error(f"Erro ao remover modelo: {exc}")
            return False

    def is_model_available(self, model_name: str) -> bool:
        """
        Verifica se um modelo está disponível localmente.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            bool: True se disponível
        """
        models = self.list_models()
        return any(model['name'] == model_name for model in models)

    def list_models(self) -> List[Dict[str, str]]:
        """
        Lista todos os modelos disponíveis localmente.
        
        Returns:
            List[Dict]: Lista de modelos
        """
        try:
            url = f"{self._api_url}/tags"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                self._logger.error("Erro ao listar modelos")
                return []
            
            data = response.json()
            models = []
            
            for model in data.get('models', []):
                model_info = {
                    'name': model.get('name', ''),
                    'size': self._format_size(model.get('size', 0)),
                    'modified_at': model.get('modified_at', ''),
                }
                models.append(model_info)
            
            return models
            
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Erro ao listar modelos: {e}")
            return []

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
            temperature: Controla aleatoriedade (0.0 a 2.0)
            max_tokens: Número máximo de tokens
            top_p: Nucleus sampling (0.0 a 1.0)
            top_k: Top-K sampling
            repeat_penalty: Penalidade por repetição

        Returns:
            str: Resposta gerada
        """
        if not self.is_model_available(model_name):
            raise ValueError(f"Modelo '{model_name}' não está disponível localmente.")

        try:
            url = f"{self._api_url}/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty,
                },
            }

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            self._logger.info(f"Chamando Ollama [{model_name}]...")

            response = requests.post(url, json=payload, timeout=180)

            if response.status_code != 200:
                raise RuntimeError(f"Ollama retornou status {response.status_code}: {response.text}")

            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            self._logger.error(f"Erro na requisição ao Ollama: {e}")
            raise RuntimeError(f"Falha na comunicação com Ollama: {e}") from e

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
        Igual a generate() mas retorna também os metadados completos da resposta Ollama.

        Returns:
            dict com chaves 'text' (str) e 'metadata' (dict) contendo tokens, durations, etc.
        """
        if not self.is_model_available(model_name):
            raise ValueError(f"Modelo '{model_name}' não está disponível localmente.")

        url = f"{self._api_url}/generate"
        payload: Dict = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(url, json=payload, timeout=180)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama retornou status {response.status_code}: {response.text}"
                )
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Falha na comunicação com Ollama: {exc}") from exc

        text = data.get("response", "").strip()

        _ns = 1_000_000_000
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        eval_dur = data.get("eval_duration")

        metadata: Dict = {
            "model": data.get("model"),
            "created_at": data.get("created_at"),
            "done_reason": data.get("done_reason"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                (prompt_tokens or 0) + (completion_tokens or 0)
            ) or None,
            "total_duration_s": (
                round(data["total_duration"] / _ns, 3)
                if data.get("total_duration")
                else None
            ),
            "load_duration_s": (
                round(data["load_duration"] / _ns, 3)
                if data.get("load_duration")
                else None
            ),
            "prompt_eval_duration_s": (
                round(data["prompt_eval_duration"] / _ns, 3)
                if data.get("prompt_eval_duration")
                else None
            ),
            "eval_duration_s": (
                round(eval_dur / _ns, 3) if eval_dur else None
            ),
        }

        if completion_tokens and metadata["eval_duration_s"] and metadata["eval_duration_s"] > 0:
            metadata["tokens_per_second"] = round(
                completion_tokens / metadata["eval_duration_s"], 1
            )

        return {"text": text, "metadata": metadata}

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Formata o tamanho em bytes para formato legível.
        Método utilitário estático (Clean Code).
        
        Args:
            size_bytes: Tamanho em bytes
            
        Returns:
            str: Tamanho formatado (ex: '1.5 GB')
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
