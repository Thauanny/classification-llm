"""
Modelos de dados do projeto.
Classes simples e focadas (Single Responsibility Principle).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Enum para os papéis nas mensagens."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    Representa uma mensagem na conversa.
    Dataclass para imutabilidade e clareza (Clean Code).
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Converte a mensagem para dicionário."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Conversation:
    """
    Representa uma conversa completa.
    Encapsula o estado da conversa (Clean Code).
    """
    id: str
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: MessageRole, content: str) -> None:
        """Adiciona uma mensagem à conversa."""
        message = Message(role=role, content=content)
        self.messages.append(message)

    def get_messages_for_api(self) -> List[dict]:
        """
        Retorna as mensagens formatadas para a API do Ollama.
        Método com nome claro e propósito único (Clean Code).
        """
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": MessageRole.SYSTEM.value,
                "content": self.system_prompt
            })
        
        for msg in self.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return messages


@dataclass
class ModelInfo:
    """
    Informações sobre um modelo LLM.
    Classe simples e focada (Single Responsibility).
    """
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None

    def to_dict(self) -> dict:
        """Converte as informações para dicionário."""
        return {
            "name": self.name,
            "size": self.size,
            "modified_at": self.modified_at,
            "digest": self.digest
        }


@dataclass
class GenerationConfig:
    """
    Configurações para geração de texto.
    Encapsula parâmetros de configuração (Clean Code).
    """
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1

    def validate(self) -> bool:
        """
        Valida as configurações.
        Método com nome claro e propósito único.
        """
        if not 0.0 <= self.temperature <= 2.0:
            return False
        if not 0.0 <= self.top_p <= 1.0:
            return False
        if self.max_tokens is not None and self.max_tokens <= 0:
            return False
        return True
