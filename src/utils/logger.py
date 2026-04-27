"""
Sistema de logging seguindo SOLID e Clean Code.
"""

import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from ..interfaces.llm_interface import ILogger


class ConsoleLogger(ILogger):
    """
    Implementação de logger para console.
    Single Responsibility: Apenas logging no console.
    """

    def __init__(self, name: str = "OllamaProject", level: int = logging.INFO):
        """
        Inicializa o logger.
        
        Args:
            name: Nome do logger
            level: Nível de logging (INFO, DEBUG, WARNING, ERROR)
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        # Remove handlers existentes para evitar duplicação
        if not self._logger.handlers:
            self._setup_console_handler()

    def _setup_console_handler(self) -> None:
        """Configura o handler para console."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Formato limpo e legível
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        """Registra uma mensagem informativa."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Registra um aviso."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Registra um erro."""
        self._logger.error(message)

    def debug(self, message: str) -> None:
        """Registra uma mensagem de debug."""
        self._logger.debug(message)


class FileLogger(ILogger):
    """
    Implementação de logger para arquivo.
    Single Responsibility: Apenas logging em arquivo.
    Open/Closed: Pode ser estendido sem modificar ILogger.
    """

    def __init__(
        self,
        name: str = "OllamaProject",
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ):
        """
        Inicializa o logger de arquivo.
        
        Args:
            name: Nome do logger
            log_file: Caminho para o arquivo de log
            level: Nível de logging
        """
        self._logger = logging.getLogger(f"{name}_file")
        self._logger.setLevel(level)
        
        if log_file is None:
            log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        self._log_file = Path(log_file)
        self._ensure_log_directory()
        
        if not self._logger.handlers:
            self._setup_file_handler()

    def _ensure_log_directory(self) -> None:
        """Cria o diretório de logs se não existir."""
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _setup_file_handler(self) -> None:
        """Configura o handler para arquivo."""
        file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self._logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Registra uma mensagem informativa."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Registra um aviso."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Registra um erro."""
        self._logger.error(message)

    def debug(self, message: str) -> None:
        """Registra uma mensagem de debug."""
        self._logger.debug(message)


class CompositeLogger(ILogger):
    """
    Logger que combina múltiplos loggers.
    Demonstra o padrão Composite e Open/Closed Principle.
    """

    def __init__(self, loggers: list[ILogger]):
        """
        Inicializa o logger composto.
        
        Args:
            loggers: Lista de loggers a serem utilizados
        """
        self._loggers = loggers

    def info(self, message: str) -> None:
        """Registra uma mensagem informativa em todos os loggers."""
        for logger in self._loggers:
            logger.info(message)

    def warning(self, message: str) -> None:
        """Registra um aviso em todos os loggers."""
        for logger in self._loggers:
            logger.warning(message)

    def error(self, message: str) -> None:
        """Registra um erro em todos os loggers."""
        for logger in self._loggers:
            logger.error(message)

    def debug(self, message: str) -> None:
        """Registra uma mensagem de debug em todos os loggers."""
        for logger in self._loggers:
            logger.debug(message)
