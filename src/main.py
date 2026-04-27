"""
Aplicação principal - Demonstração de POO, SOLID e Clean Code com Ollama.

Este módulo demonstra:
- Dependency Injection
- Inversão de Dependência
- Composição sobre Herança
- Clean Code principles
"""

import sys
from typing import Optional

from src.services.ollama_service import OllamaService
from src.services.chat_service import ChatService
from src.utils.logger import ConsoleLogger, FileLogger, CompositeLogger


class OllamaApplication:
    """
    Classe principal da aplicação.
    
    Demonstra:
    - Single Responsibility: Apenas coordenação de alto nível
    - Dependency Injection: Recebe dependências no construtor
    - Clean Code: Métodos pequenos e com propósito claro
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b-instruct-fp16"
    ):
        """
        Inicializa a aplicação.
        
        Args:
            model_name: Nome do modelo Ollama a ser usado
        """
        self._model_name = model_name
        self._setup_logger()
        self._setup_services()

    def _setup_logger(self) -> None:
        """Configura o sistema de logging."""
        console_logger = ConsoleLogger(name="OllamaApp")
        file_logger = FileLogger(name="OllamaApp")
        
        # Usa CompositeLogger para log em múltiplos destinos
        self._logger = CompositeLogger([console_logger, file_logger])
        self._logger.info("=== Ollama Application Iniciada ===")

    def _setup_services(self) -> None:
        """
        Configura os serviços da aplicação.
        Demonstra Dependency Injection e Inversão de Dependência.
        """
        # Cria serviço Ollama (injeção de dependência)
        self._ollama_service = OllamaService(logger=self._logger)
        
        # Cria serviço de chat (composição)
        self._chat_service = ChatService(
            llm_service=self._ollama_service,
            logger=self._logger,
            model_name=self._model_name
        )

    def ensure_model_downloaded(self) -> bool:
        """
        Garante que o modelo está baixado.
        Método com nome claro e propósito único.
        
        Returns:
            bool: True se o modelo está disponível
        """
        self._logger.info(f"Verificando modelo: {self._model_name}")
        
        if self._ollama_service.is_model_available(self._model_name):
            self._logger.info(f"Modelo {self._model_name} já está disponível!")
            return True
        
        self._logger.info(f"Modelo não encontrado. Iniciando download...")
        print(f"\n⏬ Baixando modelo {self._model_name}...")
        print("Isso pode levar alguns minutos dependendo da sua conexão.\n")
        
        success = self._ollama_service.pull_model(self._model_name)
        
        if success:
            print(f"\n✅ Modelo {self._model_name} baixado com sucesso!\n")
        else:
            print(f"\n❌ Falha ao baixar o modelo {self._model_name}\n")
            print("Certifique-se de que o Ollama está rodando: ollama serve\n")
        
        return success

    def list_available_models(self) -> None:
        """Lista todos os modelos disponíveis."""
        self._logger.info("Listando modelos disponíveis...")
        models = self._ollama_service.list_models()
        
        if not models:
            print("Nenhum modelo encontrado.")
            return
        
        print("\n📚 Modelos disponíveis:")
        print("-" * 60)
        for model in models:
            print(f"  • {model['name']}")
            print(f"    Tamanho: {model['size']}")
            print(f"    Modificado: {model['modified_at']}")
            print()

    def run_interactive_chat(self) -> None:
        """
        Executa um chat interativo.
        Método principal de interação com o usuário.
        """
        if not self.ensure_model_downloaded():
            return

        print("\n" + "=" * 60)
        print("🤖 Chat Interativo com Ollama")
        print("=" * 60)
        print(f"Modelo: {self._model_name}")
        print("\nComandos especiais:")
        print("  /sair - Encerra o chat")
        print("  /novo - Inicia uma nova conversa")
        print("  /historico - Mostra o histórico da conversa")
        print("  /modelos - Lista modelos disponíveis")
        print("=" * 60 + "\n")

        # Inicia uma conversa
        system_prompt = (
            "Você é um assistente útil e amigável. "
            "Responda de forma clara e concisa."
        )
        conversation_id = self._chat_service.start_conversation(system_prompt)

        self._run_chat_loop(conversation_id)

    def _run_chat_loop(self, conversation_id: str) -> None:
        """
        Loop principal do chat.
        Método privado que encapsula a lógica do loop.
        
        Args:
            conversation_id: ID da conversa atual
        """
        current_conv_id = conversation_id
        
        while True:
            try:
                user_input = input("\n👤 Você: ").strip()
                
                if not user_input:
                    continue

                # Processa comandos especiais
                if user_input.startswith('/'):
                    current_conv_id = self._handle_command(
                        user_input,
                        current_conv_id
                    )
                    if current_conv_id is None:
                        break
                    continue

                # Envia mensagem e obtém resposta
                print("\n🤖 Assistente: ", end="", flush=True)
                response = self._chat_service.send_message(
                    current_conv_id,
                    user_input
                )
                print(response)

            except KeyboardInterrupt:
                print("\n\nChat interrompido pelo usuário.")
                break
            except Exception as e:
                self._logger.error(f"Erro no chat: {e}")
                print(f"\n❌ Erro: {e}")

        self._logger.info("Chat encerrado")

    def _handle_command(
        self,
        command: str,
        current_conversation_id: str
    ) -> Optional[str]:
        """
        Processa comandos especiais.
        Método com responsabilidade única (Clean Code).
        
        Args:
            command: Comando a ser processado
            current_conversation_id: ID da conversa atual
            
        Returns:
            Optional[str]: Novo ID de conversa ou None para sair
        """
        command = command.lower()

        if command == '/sair':
            print("\n👋 Até logo!")
            return None

        elif command == '/novo':
            system_prompt = (
                "Você é um assistente útil e amigável. "
                "Responda de forma clara e concisa."
            )
            new_id = self._chat_service.start_conversation(system_prompt)
            print("\n✨ Nova conversa iniciada!")
            return new_id

        elif command == '/historico':
            self._show_history(current_conversation_id)
            return current_conversation_id

        elif command == '/modelos':
            self.list_available_models()
            return current_conversation_id

        else:
            print(f"\n❌ Comando desconhecido: {command}")
            return current_conversation_id

    def _show_history(self, conversation_id: str) -> None:
        """
        Mostra o histórico da conversa.
        
        Args:
            conversation_id: ID da conversa
        """
        history = self._chat_service.get_history(conversation_id)
        
        if not history:
            print("\n📝 Histórico vazio")
            return

        print("\n" + "=" * 60)
        print("📝 Histórico da Conversa")
        print("=" * 60)
        
        for msg in history:
            role = "👤 Você" if msg['role'] == 'user' else "🤖 Assistente"
            print(f"\n{role}:")
            print(msg['content'])
        
        print("\n" + "=" * 60)

    def run_single_query(self, query: str) -> str:
        """
        Executa uma única consulta.
        Útil para testes e automação.
        
        Args:
            query: Pergunta a ser feita
            
        Returns:
            str: Resposta do modelo
        """
        if not self.ensure_model_downloaded():
            return "Modelo não disponível"

        conversation_id = self._chat_service.start_conversation()
        response = self._chat_service.send_message(conversation_id, query)
        
        return response


def main():
    """
    Função principal de entrada.
    Demonstra a criação e uso da aplicação.
    """
    # Modelo padrão: Llama 3.2 3B Instruct
    # Outros modelos populares: "llama3.1:8b", "mistral:7b", etc.
    app = OllamaApplication(model_name="llama3.2:3b-instruct-fp16")
    
    # Verifica argumentos de linha de comando
    if len(sys.argv) > 1:
        # Modo de consulta única
        query = " ".join(sys.argv[1:])
        response = app.run_single_query(query)
        print(f"\n🤖 Resposta: {response}\n")
    else:
        # Modo interativo
        app.run_interactive_chat()


if __name__ == "__main__":
    main()
