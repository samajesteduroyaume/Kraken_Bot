import logging
import logging.config
import sys
from typing import Dict
import os
import psutil
import traceback

# Configuration du logging
# Configuration du logging avec variables d'environnement
LOGGING_CONFIG: Dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(process)d - %(threadName)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': 'json_log_formatter.JSONFormatter',
            'format': '%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(process)d - %(threadName)s - %(processName)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'performance': {
            'format': '%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'formatter': 'default',
            'stream': sys.stdout
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': os.getenv('LOG_FILE_LEVEL', 'DEBUG'),
            'formatter': 'detailed',
            'filename': os.path.expanduser(os.getenv('LOG_FILE', '~/kraken_bot_logs/kraken_bot.log')),
            'when': 'midnight',
            'interval': 1,
            'backupCount': int(os.getenv('LOG_BACKUP_COUNT', '7')),
            'encoding': 'utf-8'
        },
        'detailed': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': os.getenv('LOG_FILE_LEVEL', 'DEBUG'),
            'formatter': 'detailed',
            'filename': os.path.expanduser(os.getenv('LOG_DETAILED_FILE', '~/kraken_bot_logs/kraken_bot_detailed.log')),
            # 10MB par défaut
            'maxBytes': int(os.getenv('LOG_MAX_SIZE', '10485760')),
            'backupCount': int(os.getenv('LOG_BACKUP_COUNT', '10')),
            'encoding': 'utf-8'
        },
        'json': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': os.getenv('LOG_FILE_LEVEL', 'DEBUG'),
            'formatter': 'json',
            'filename': os.path.expanduser(os.getenv('LOG_JSON_FILE', '~/kraken_bot_logs/kraken_bot.json.log')),
            'when': 'midnight',
            'interval': 1,
            'backupCount': int(os.getenv('LOG_BACKUP_COUNT', '7')),
            'encoding': 'utf-8'
        },
        'error': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': os.path.expanduser(os.getenv('LOG_ERROR_FILE', '~/kraken_bot_logs/kraken_bot_errors.log')),
            # 5MB par défaut
            'maxBytes': int(os.getenv('LOG_ERROR_MAX_SIZE', '5242880')),
            'backupCount': int(os.getenv('LOG_ERROR_BACKUP_COUNT', '5')),
            'encoding': 'utf-8'
        },
        'performance': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': os.getenv('LOG_PERFORMANCE_LEVEL', 'INFO'),
            'formatter': 'performance',
            'filename': os.path.expanduser(os.getenv('LOG_PERFORMANCE_FILE', '~/kraken_bot_logs/kraken_bot_performance.log')),
            'when': 'midnight',
            'interval': 1,
            'backupCount': int(os.getenv('LOG_PERFORMANCE_BACKUP_COUNT', '7')),
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file', 'detailed', 'json', 'error', 'performance'],
            'level': os.getenv('LOG_LEVEL', 'DEBUG'),
            'propagate': True
        },
        'src': {
            'handlers': ['console', 'file', 'detailed', 'json', 'error', 'performance'],
            'level': 'INFO',
            'propagate': False
        },
        'aiohttp': {
            'handlers': ['console', 'file', 'detailed', 'json', 'error'],
            'level': 'WARNING',
            'propagate': False
        },
        'asyncio': {
            'handlers': ['console', 'file', 'detailed', 'json', 'error'],
            'level': 'WARNING',
            'propagate': False
        }
    }
}


def setup_logging():
    """Configure le système de logging."""
    try:
        # Création des dossiers de logs
        log_dir = '/tmp'
        os.makedirs(log_dir, exist_ok=True)

        # Configuration du logging
        logging.config.dictConfig(LOGGING_CONFIG)
        logger = logging.getLogger(__name__)
        logger.info("Configuration du logging initialisée avec succès")

        # Configuration du logger de performance
        performance_logger = logging.getLogger('performance')
        performance_logger.setLevel(logging.INFO)

        # Ajout du filtre de métriques
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                record.cpu_usage = psutil.cpu_percent()
                record.memory_usage = psutil.virtual_memory().percent
                return True

        performance_logger.addFilter(PerformanceFilter())

        return logger
    except Exception as e:
        print(f"Erreur lors de la configuration du logging: {e}")
        raise

# Couleurs pour les logs


class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs, emojis et styles pour les logs"""
    COLORS = {
        'DEBUG': '\033[94m',    # Bleu
        'INFO': '\033[92m',     # Vert
        'WARNING': '\033[93m',  # Jaune
        'ERROR': '\033[91m',    # Rouge
        'CRITICAL': '\033[95m'  # Magenta
    }

    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }

    STYLES = {
        'DEBUG': '\033[1m',     # Gras
        'INFO': '\033[1m',      # Gras
        'WARNING': '\033[1m',   # Gras
        'ERROR': '\033[1m',     # Gras
        'CRITICAL': '\033[1m'   # Gras
    }

    def format(self, record):
        """Formatte le message de log avec des informations supplémentaires."""
        color = self.COLORS.get(record.levelname, '')
        emoji = self.EMOJIS.get(record.levelname, '')
        style = self.STYLES.get(record.levelname, '')
        reset = '\033[0m'
        formatted = super().format(record)

        # Ajouter des détails supplémentaires pour les erreurs
        if record.levelname in ['ERROR', 'CRITICAL']:
            exc_info = getattr(record, 'exc_info', None)
            if exc_info:
                tb = ''.join(traceback.format_exception(*exc_info))
                extra_details = f'\n{tb}'
            else:
                extra_details = ''
        else:
            extra_details = ''

        # Ajouter le nom du fichier et le numéro de ligne pour les erreurs
        if record.levelname in ['ERROR', 'CRITICAL']:
            source_info = f' [{record.filename}:{record.lineno}]'
        else:
            source_info = ''

        # Ajouter des informations supplémentaires
        extra_info = []
        if hasattr(record, 'duration'):
            extra_info.append(f'[{record.duration:.2f}s]')
        if hasattr(record, 'pair'):
            extra_info.append(f'[{record.pair}]')

        extra_info_str = ' '.join(extra_info) if extra_info else ''

        # Format final
        msg = f"{color}{emoji}{style}{formatted}{reset}{source_info} {extra_info_str}{extra_details}"

        return msg
        if hasattr(record, 'type'):
            extra_info.append(f'[{record.type}]')
        extra_info_str = ' '.join(extra_info) if extra_info else ''

        return f'{color}{style}{emoji} {super().format(record)} {extra_info_str}{source_info}{extra_details}{reset}'


def setup_colored_logging():
    """Configure le logging avec des couleurs et des emojis"""
    try:
        # Configurer le logger principal
        logger = logging.getLogger()

        # Créer un handler console avec le formatter coloré
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        ))

        # Ajouter le handler console
        logger.addHandler(console_handler)

        # Configurer le niveau de logging
        logger.setLevel(logging.INFO)

        # Ajouter des formatters spécifiques pour les niveaux de log
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.addFilter(lambda record: record.levelname != 'DEBUG')

        return logger
    except Exception as e:
        print(f"❌ Erreur lors de la configuration du logging coloré: {e}")
        raise


# Initialiser le logging
setup_logging()
setup_colored_logging()
