"""
Exemplo de testes unitários.
Demonstra como testar o código seguindo SOLID principles.
"""

import unittest
from unittest.mock import Mock, patch
from src.models.llm_model import Message, Conversation, MessageRole, GenerationConfig


class TestMessage(unittest.TestCase):
    """Testes para a classe Message."""

    def test_message_creation(self):
        """Testa a criação de uma mensagem."""
        message = Message(
            role=MessageRole.USER,
            content="Olá, mundo!"
        )
        
        self.assertEqual(message.role, MessageRole.USER)
        self.assertEqual(message.content, "Olá, mundo!")
        self.assertIsNotNone(message.timestamp)

    def test_message_to_dict(self):
        """Testa a conversão de mensagem para dicionário."""
        message = Message(
            role=MessageRole.ASSISTANT,
            content="Olá! Como posso ajudar?"
        )
        
        msg_dict = message.to_dict()
        
        self.assertEqual(msg_dict['role'], 'assistant')
        self.assertEqual(msg_dict['content'], "Olá! Como posso ajudar?")
        self.assertIn('timestamp', msg_dict)


class TestConversation(unittest.TestCase):
    """Testes para a classe Conversation."""

    def test_conversation_creation(self):
        """Testa a criação de uma conversa."""
        conv = Conversation(id="test-123")
        
        self.assertEqual(conv.id, "test-123")
        self.assertEqual(len(conv.messages), 0)
        self.assertIsNone(conv.system_prompt)

    def test_add_message(self):
        """Testa a adição de mensagens."""
        conv = Conversation(id="test-123")
        
        conv.add_message(MessageRole.USER, "Olá")
        conv.add_message(MessageRole.ASSISTANT, "Oi!")
        
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].role, MessageRole.USER)
        self.assertEqual(conv.messages[1].role, MessageRole.ASSISTANT)

    def test_get_messages_for_api(self):
        """Testa a formatação de mensagens para a API."""
        conv = Conversation(
            id="test-123",
            system_prompt="Você é um assistente útil."
        )
        
        conv.add_message(MessageRole.USER, "Olá")
        conv.add_message(MessageRole.ASSISTANT, "Oi!")
        
        api_messages = conv.get_messages_for_api()
        
        # Deve ter system prompt + 2 mensagens
        self.assertEqual(len(api_messages), 3)
        self.assertEqual(api_messages[0]['role'], 'system')
        self.assertEqual(api_messages[1]['role'], 'user')
        self.assertEqual(api_messages[2]['role'], 'assistant')


class TestGenerationConfig(unittest.TestCase):
    """Testes para a classe GenerationConfig."""

    def test_default_config(self):
        """Testa configuração padrão."""
        config = GenerationConfig()
        
        self.assertEqual(config.temperature, 0.7)
        self.assertIsNone(config.max_tokens)
        self.assertTrue(config.validate())

    def test_valid_config(self):
        """Testa configuração válida."""
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=100
        )
        
        self.assertTrue(config.validate())

    def test_invalid_temperature(self):
        """Testa temperatura inválida."""
        config = GenerationConfig(temperature=3.0)
        self.assertFalse(config.validate())
        
        config = GenerationConfig(temperature=-0.5)
        self.assertFalse(config.validate())

    def test_invalid_max_tokens(self):
        """Testa max_tokens inválido."""
        config = GenerationConfig(max_tokens=-10)
        self.assertFalse(config.validate())


class TestLogger(unittest.TestCase):
    """Testes para o sistema de logging."""

    def test_console_logger_interface(self):
        """Testa se ConsoleLogger implementa ILogger."""
        from src.utils.logger import ConsoleLogger
        from src.interfaces.llm_interface import ILogger
        
        logger = ConsoleLogger()
        self.assertIsInstance(logger, ILogger)

    def test_logger_methods_exist(self):
        """Testa se todos os métodos de log existem."""
        from src.utils.logger import ConsoleLogger
        
        logger = ConsoleLogger()
        
        # Verifica se os métodos existem
        self.assertTrue(hasattr(logger, 'info'))
        self.assertTrue(hasattr(logger, 'warning'))
        self.assertTrue(hasattr(logger, 'error'))
        self.assertTrue(hasattr(logger, 'debug'))


class TestOllamaService(unittest.TestCase):
    """Testes para OllamaService."""

    def setUp(self):
        """Configuração antes de cada teste."""
        from src.utils.logger import ConsoleLogger
        self.logger = ConsoleLogger()

    def test_service_initialization(self):
        """Testa a inicialização do serviço."""
        from src.services.ollama_service import OllamaService
        
        service = OllamaService(logger=self.logger)
        self.assertIsNotNone(service)

    def test_format_size(self):
        """Testa a formatação de tamanho."""
        from src.services.ollama_service import OllamaService
        
        # Testa diferentes tamanhos
        self.assertEqual(OllamaService._format_size(1024), "1.00 KB")
        self.assertEqual(OllamaService._format_size(1024 * 1024), "1.00 MB")
        self.assertEqual(OllamaService._format_size(1024 * 1024 * 1024), "1.00 GB")


class TestChatService(unittest.TestCase):
    """Testes para ChatService."""

    def setUp(self):
        """Configuração antes de cada teste."""
        from src.utils.logger import ConsoleLogger
        from src.interfaces.llm_interface import ILLMService
        
        # Cria um mock do LLM service
        self.mock_llm = Mock(spec=ILLMService)
        self.logger = ConsoleLogger()

    def test_service_initialization(self):
        """Testa a inicialização do ChatService."""
        from src.services.chat_service import ChatService
        
        service = ChatService(
            llm_service=self.mock_llm,
            logger=self.logger,
            model_name="test-model"
        )
        
        self.assertIsNotNone(service)

    def test_start_conversation(self):
        """Testa o início de uma conversa."""
        from src.services.chat_service import ChatService
        
        service = ChatService(
            llm_service=self.mock_llm,
            logger=self.logger,
            model_name="test-model"
        )
        
        conv_id = service.start_conversation()
        self.assertIsNotNone(conv_id)
        self.assertIsInstance(conv_id, str)

    def test_list_conversations(self):
        """Testa a listagem de conversas."""
        from src.services.chat_service import ChatService
        
        service = ChatService(
            llm_service=self.mock_llm,
            logger=self.logger,
            model_name="test-model"
        )
        
        # Inicia algumas conversas
        conv_id1 = service.start_conversation()
        conv_id2 = service.start_conversation()
        
        conversations = service.list_conversations()
        self.assertEqual(len(conversations), 2)


def run_tests():
    """Executa todos os testes."""
    unittest.main()


if __name__ == '__main__':
    run_tests()
