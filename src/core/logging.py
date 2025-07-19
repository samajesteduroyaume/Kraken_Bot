"""
Module de logging pour le bot de trading Kraken.
Gère les logs, les alertes et la rotation des fichiers de log.
"""

import os
import logging
from typing import Dict
from datetime import datetime
import os
from pathlib import Path


class LoggerManager:
    """Gestionnaire de logging pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de logging."""
        self.config = config
        self.log_dir = config.get('log_dir', 'logs')
        self.log_level = config.get('log_level', 'INFO').upper()
        self.max_bytes = config.get('max_bytes', 10485760)  # 10MB par défaut
        self.backup_count = config.get('backup_count', 5)

        # Créer le répertoire de logs s'il n'existe pas
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Configurer le logger root
        self.setup_root_logger()

    def setup_root_logger(self) -> None:
        """Configure le logger root avec les handlers appropriés."""
        try:
            # Créer le logger
            logger = logging.getLogger()
            logger.setLevel(self.log_level)

            # Créer le formatteur
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Handler console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Handler fichier avec rotation
            log_file = os.path.join(
                self.log_dir,
                f"kraken_bot_{datetime.now().strftime('%Y%m%d')}.log"
            )

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            print(f"Erreur lors de la configuration du logger: {str(e)}")
            raise

    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtient un logger nommé.

        Args:
            name: Nom du logger

        Returns:
            Logger configuré
        """
        return logging.getLogger(name)

    def set_level(self, level: str) -> None:
        """
        Définit le niveau de logging.

        Args:
            level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            level = level.upper()
            if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ValueError(f"Niveau de logging invalide: {level}")

            self.log_level = level
            logging.getLogger().setLevel(level)

        except Exception as e:
            print(f"Erreur lors du changement de niveau de logging: {str(e)}")
            raise
