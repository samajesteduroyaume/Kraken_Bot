import os
import logging
import tarfile
import datetime
from pathlib import Path
from typing import Dict, Any
from src.core.config_adapter import Config  # Migré vers le nouvel adaptateur
import boto3
from botocore.exceptions import NoCredentialsError

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogBackup:
    """Système de backup des logs."""

    def __init__(self, config: Config):
        """Initialise le système de backup."""
        self.config = config
        log_config = config.get('log_config', {})
        backup_config = config.get('backup_config', {})
        
        self.log_dir = os.path.expanduser(
            log_config.get('dir', '~/kraken_bot_logs'))
        self.backup_dir = os.path.expanduser(
            backup_config.get('dir', '~/kraken_bot_backups'))
        self.backup_interval = backup_config.get('interval', 'daily')  # daily, weekly, monthly
        self.aws_config = config.get('aws_config', {})

        # Créer le répertoire de backup s'il n'existe pas
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)

    def _get_backup_filename(self) -> str:
        """Génère le nom du fichier de backup."""
        now = datetime.datetime.now()
        if self.backup_interval == 'daily':
            return f"logs_backup_{now.strftime('%Y%m%d')}.tar.gz"
        elif self.backup_interval == 'weekly':
            return f"logs_backup_{now.strftime('%Y%W')}.tar.gz"
        elif self.backup_interval == 'monthly':
            return f"logs_backup_{now.strftime('%Y%m')}.tar.gz"
        return f"logs_backup_{now.strftime('%Y%m%d')}.tar.gz"

    def _compress_logs(self) -> str:
        """Compresse les fichiers de log en un fichier tar.gz."""
        try:
            backup_file = os.path.join(
                self.backup_dir, self._get_backup_filename())

            with tarfile.open(backup_file, 'w:gz') as tar:
                for file in Path(self.log_dir).glob('*'):
                    tar.add(file, arcname=file.name)

            logger.info(f"Logs compressés dans {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"Erreur lors de la compression des logs: {str(e)}")
            return None

    def _upload_to_s3(self, backup_file: str) -> bool:
        """Upload le fichier de backup vers S3."""
        try:
            if not self.aws_config.get('enabled', False):
                return False

            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_config['access_key'],
                aws_secret_access_key=self.aws_config['secret_key']
            )

            bucket = self.aws_config['bucket']
            key = f"logs/{os.path.basename(backup_file)}"

            s3.upload_file(backup_file, bucket, key)
            logger.info(f"Backup uploadé vers S3: s3://{bucket}/{key}")
            return True

        except NoCredentialsError:
            logger.error("Aucune credentials AWS trouvées")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de l'upload vers S3: {str(e)}")
            return False

    def _rotate_backups(self):
        """Gère la rotation des backups locaux."""
        try:
            max_backups = self.config.get('MAX_BACKUPS', 10)
            backups = sorted(Path(self.backup_dir).glob('*.tar.gz'))

            if len(backups) > max_backups:
                for backup in backups[:-max_backups]:
                    backup.unlink()
                    logger.info(f"Backup supprimé: {backup.name}")

        except Exception as e:
            logger.error(f"Erreur lors de la rotation des backups: {str(e)}")

    def backup_logs(self) -> bool:
        """Effectue un backup complet des logs."""
        try:
            # Créer le backup
            backup_file = self._compress_logs()
            if not backup_file:
                return False

            # Upload vers S3 si configuré
            if self.aws_config.get('enabled', False):
                self._upload_to_s3(backup_file)

            # Gérer la rotation des backups
            self._rotate_backups()

            logger.info("Backup des logs terminé avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur lors du backup des logs: {str(e)}")
            return False

    def restore_backup(self, backup_file: str) -> bool:
        """Restaure un backup des logs."""
        try:
            if not os.path.exists(backup_file):
                logger.error(f"Fichier de backup {backup_file} non trouvé")
                return False

            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(path=self.log_dir)

            logger.info(f"Backup restauré depuis {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la restauration du backup: {str(e)}")
            return False


def setup_log_backup(config: Dict[str, Any]) -> LogBackup:
    """Configure le système de backup des logs."""
    backup = LogBackup(config)
    return backup


def backup_logs(config: Dict[str, Any]) -> bool:
    """Effectue un backup des logs selon la configuration."""
    backup = setup_log_backup(config)
    return backup.backup_logs()


def restore_logs(config: Dict[str, Any], backup_file: str) -> bool:
    """Restaure les logs à partir d'un backup."""
    backup = setup_log_backup(config)
    return backup.restore_backup(backup_file)
