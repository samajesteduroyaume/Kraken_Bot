"""
Module de gestion des logs pour le bot de trading.
"""
import logging
from typing import Optional
import sys


class LoggerManager:
    """Gestionnaire de logs pour le bot de trading."""

    def __init__(self,
                 name: str = 'trading_bot',
                 level: str = 'DEBUG',
                 log_file: Optional[str] = None,
                 console: bool = True,
                 rotation: bool = True,
                 max_bytes: int = 10485760,
                 backup_count: int = 5):
        """
        Initialise le gestionnaire de logs.

        Args:
            name: Nom du logger
            level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Chemin du fichier de log
            console: Si True, affiche les logs dans la console
            rotation: Si True, active la rotation des logs
            max_bytes: Taille maximale d'un fichier de log en bytes
            backup_count: Nombre de fichiers de log à conserver
        """
        self.name = name
        self.level = level.upper()
        self.log_file = log_file
        self.console = console
        self.rotation = rotation
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Créer le logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Configurer les formateurs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Configurer le handler console
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Configurer le handler fichier
        if self.log_file:
            if self.rotation:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count
                )
            else:
                file_handler = logging.FileHandler(self.log_file)

            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Retourne le logger configuré ou un logger nommé."""
        if name is None or name == self.name:
            return self.logger
        else:
            return logging.getLogger(name)

    def set_level(self, level: str):
        """Définit le niveau de log."""
        self.level = level.upper()
        self.logger.setLevel(self.level)

        # Mettre à jour les niveaux des handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.level)

    def log(self, level: str, message: str):
        """Log un message avec le niveau spécifié."""
        level = level.upper()
        if level == 'DEBUG':
            self.logger.debug(message)
        elif level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        elif level == 'CRITICAL':
            self.logger.critical(message)
        else:
            self.logger.info(message)

    def debug(self, message: str):
        """Log un message de debug."""
        self.log('DEBUG', message)

    def info(self, message: str):
        """Log un message d'information."""
        self.log('INFO', message)

    def warning(self, message: str):
        """Log un message d'avertissement."""
        self.log('WARNING', message)

    def error(self, message: str):
        """Log un message d'erreur."""
        self.log('ERROR', message)

    def critical(self, message: str):
        """Log un message critique."""
        self.log('CRITICAL', message)
