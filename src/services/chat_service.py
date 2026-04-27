"""
Serviço de chat usando Ollama.
Demonstra composição e delegation (SOLID).
"""

import uuid
from typing import Dict, List, Optional

from ..interfaces.llm_interface import IChatService, ILLMService, ILogger
from ..models.llm_model import Conversation, MessageRole


class ChatService(IChatService):
    """
    Serviço de chat que gerencia conversas.
    
    Princípios SOLID:
    - Single Responsibility: Apenas gerenciamento de conversas
    - Dependency Inversion: Depende de abstrações (ILLMService, ILogger)
    - Open/Closed: Extensível via herança
    """

    def __init__(
        self,
        llm_service: ILLMService,
        logger: ILogger,
        model_name: str
    ):
        """
        Inicializa o serviço de chat.
        
        Args:
            llm_service: Serviço de LLM (Dependency Injection)
            logger: Logger para eventos
            model_name: Nome do modelo a ser usado
        """
        self._llm_service = llm_service
        self._logger = logger
        self._model_name = model_name
        self._conversations: Dict[str, Conversation] = {}
        
        self._logger.info(f"ChatService inicializado com modelo: {model_name}")

    def start_conversation(self, system_prompt: Optional[str] = None) -> str:
        """
        Inicia uma nova conversa.
        
        Args:
            system_prompt: Prompt de sistema opcional
            
        Returns:
            str: ID único da conversa
        """
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            system_prompt=system_prompt
        )
        
        self._conversations[conversation_id] = conversation
        self._logger.info(f"Nova conversa iniciada: {conversation_id}")
        
        return conversation_id

    def send_message(self, conversation_id: str, message: str) -> str:
        """
        Envia uma mensagem e obtém resposta.
        
        Args:
            conversation_id: ID da conversa
            message: Mensagem do usuário
            
        Returns:
            str: Resposta do assistente
        """
        if not self._is_valid_conversation(conversation_id):
            error_msg = f"Conversa {conversation_id} não encontrada"
            self._logger.error(error_msg)
            return error_msg

        conversation = self._conversations[conversation_id]
        
        # Adiciona mensagem do usuário
        conversation.add_message(MessageRole.USER, message)
        self._logger.debug(f"Mensagem do usuário: {message[:50]}...")

        # Gera resposta usando o LLM
        response = self._generate_response(conversation)
        
        # Adiciona resposta do assistente
        conversation.add_message(MessageRole.ASSISTANT, response)
        self._logger.debug(f"Resposta do assistente: {response[:50]}...")

        return response

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Obtém o histórico de uma conversa.
        
        Args:
            conversation_id: ID da conversa
            
        Returns:
            List[Dict]: Histórico formatado
        """
        if not self._is_valid_conversation(conversation_id):
            self._logger.error(f"Conversa {conversation_id} não encontrada")
            return []

        conversation = self._conversations[conversation_id]
        return [msg.to_dict() for msg in conversation.messages]

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Deleta uma conversa.
        
        Args:
            conversation_id: ID da conversa
            
        Returns:
            bool: True se deletada com sucesso
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._logger.info(f"Conversa deletada: {conversation_id}")
            return True
        
        self._logger.warning(f"Tentativa de deletar conversa inexistente: {conversation_id}")
        return False

    def list_conversations(self) -> List[Dict[str, str]]:
        """
        Lista todas as conversas ativas.
        
        Returns:
            List[Dict]: Lista de conversas com informações básicas
        """
        conversations = []
        for conv_id, conv in self._conversations.items():
            conversations.append({
                'id': conv_id,
                'messages_count': len(conv.messages),
                'created_at': conv.created_at.isoformat()
            })
        return conversations

    def _is_valid_conversation(self, conversation_id: str) -> bool:
        """
        Valida se uma conversa existe.
        Método privado auxiliar (Clean Code).
        
        Args:
            conversation_id: ID da conversa
            
        Returns:
            bool: True se válida
        """
        return conversation_id in self._conversations

    def _generate_response(self, conversation: Conversation) -> str:
        """
        Gera uma resposta usando o histórico da conversa.
        Método privado que encapsula lógica de geração (Clean Code).
        
        Args:
            conversation: Objeto da conversa
            
        Returns:
            str: Resposta gerada
        """
        # Constrói o prompt com todo o contexto
        prompt = self._build_prompt(conversation)
        
        # Usa o serviço LLM para gerar resposta
        response = self._llm_service.generate(
            prompt=prompt,
            model_name=self._model_name,
            temperature=0.7
        )
        
        return response

    def _build_prompt(self, conversation: Conversation) -> str:
        """
        Constrói o prompt com o contexto da conversa.
        Método com propósito único e nome claro (Clean Code).
        
        Args:
            conversation: Objeto da conversa
            
        Returns:
            str: Prompt completo
        """
        parts = []
        
        # Adiciona system prompt se existir
        if conversation.system_prompt:
            parts.append(f"Sistema: {conversation.system_prompt}\n")
        
        # Adiciona histórico de mensagens
        for msg in conversation.messages:
            role = "Usuário" if msg.role == MessageRole.USER else "Assistente"
            parts.append(f"{role}: {msg.content}")
        
        # Adiciona indicador para nova resposta
        parts.append("Assistente:")
        
        return "\n".join(parts)
