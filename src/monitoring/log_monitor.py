import os
import time
import logging
import threading
from typing import Dict
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogMonitor:
    """Moniteur de logs pour le bot de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Initialise le moniteur de logs."""
        self.config = config
        self.running = False
        self.threads = []
        self.alert_thresholds = {
            'error_threshold': 10,  # Nombre d'erreurs avant alerte
            'warning_threshold': 20,  # Nombre d'avertissements avant alerte
            'disk_threshold': 90,  # Pourcentage d'espace disque avant alerte
        }
        self.error_count = 0
        self.warning_count = 0
        self.last_alert_time = None
        self.alert_cooldown = 300  # 5 minutes entre les alertes

    def _monitor_logs(self):
        """Surveille les fichiers de log pour les erreurs et avertissements."""
        while self.running:
            try:
                # Vérifier les logs d'erreur
                error_log_path = os.path.expanduser(
                    self.config.get(
                        'LOG_ERROR_FILE',
                        '~/kraken_bot_logs/kraken_bot_errors.log'))
                if os.path.exists(error_log_path):
                    with open(error_log_path, 'r') as f:
                        for line in f:
                            if 'ERROR' in line:
                                self.error_count += 1
                                self._check_alerts()

                # Vérifier les logs de warning
                log_path = os.path.expanduser(self.config.get(
                    'LOG_FILE', '~/kraken_bot_logs/kraken_bot.log'))
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        for line in f:
                            if 'WARNING' in line:
                                self.warning_count += 1
                                self._check_alerts()

                # Vérifier l'espace disque
                self._check_disk_space()

                time.sleep(60)  # Vérifier toutes les minutes

            except Exception as e:
                logger.error(
                    f"Erreur lors de la surveillance des logs: {str(e)}")
                time.sleep(60)

    def _check_alerts(self):
        """Vérifie si des alertes doivent être envoyées."""
        current_time = time.time()
        if self.last_alert_time and (
                current_time - self.last_alert_time) < self.alert_cooldown:
            return

        alerts = []

        # Vérifier les erreurs
        if self.error_count >= self.alert_thresholds['error_threshold']:
            alerts.append(f"Nombre d'erreurs élevé: {self.error_count}")
            self.error_count = 0  # Réinitialiser après alerte

        # Vérifier les avertissements
        if self.warning_count >= self.alert_thresholds['warning_threshold']:
            alerts.append(
                f"Nombre d'avertissements élevé: {self.warning_count}")
            self.warning_count = 0  # Réinitialiser après alerte

        if alerts:
            self._send_alert("\n".join(alerts))
            self.last_alert_time = current_time

    def _check_disk_space(self):
        """Vérifie l'espace disque disponible."""
        try:
            total, used, free = shutil.disk_usage(
                os.path.expanduser('~/kraken_bot_logs'))
            usage_percent = (used / total) * 100

            if usage_percent >= self.alert_thresholds['disk_threshold']:
                self._send_alert(
                    f"Espace disque faible: {usage_percent:.1f}% utilisé")
        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification de l'espace disque: {str(e)}")

    def _send_alert(self, message: str):
        """Envoie une alerte."""
        try:
            # Ici, vous pouvez implémenter l'envoi d'alertes (email, Telegram,
            # etc.)
            logger.warning(f"ALERT: {message}")

            # Exemple d'alerte par email (à implémenter)
            # send_email("Alerte Kraken Bot", message)

            # Exemple d'alerte Telegram (à implémenter)
            # send_telegram_alert(message)

        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")

    def start(self):
        """Démarre le moniteur de logs."""
        self.running = True
        thread = threading.Thread(target=self._monitor_logs, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.info("Moniteur de logs démarré")

    def stop(self):
        """Arrête le moniteur de logs."""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5)
        logger.info("Moniteur de logs arrêté")


def setup_log_monitor(config: Dict[str, Any]) -> LogMonitor:
    """Configure et démarre le moniteur de logs."""
    monitor = LogMonitor(config)
    monitor.start()
    return monitor


def check_log_files(config: Dict[str, Any]) -> Dict[str, bool]:
    """Vérifie l'existence et la taille des fichiers de log."""
    log_files = {
        'main': os.path.expanduser(
            config.get(
                'LOG_FILE',
                '~/kraken_bot_logs/kraken_bot.log')),
        'detailed': os.path.expanduser(
            config.get(
                'LOG_DETAILED_FILE',
                '~/kraken_bot_logs/kraken_bot_detailed.log')),
        'json': os.path.expanduser(
            config.get(
                'LOG_JSON_FILE',
                '~/kraken_bot_logs/kraken_bot.json.log')),
        'error': os.path.expanduser(
            config.get(
                'LOG_ERROR_FILE',
                '~/kraken_bot_logs/kraken_bot_errors.log')),
        'performance': os.path.expanduser(
            config.get(
                'LOG_PERFORMANCE_FILE',
                '~/kraken_bot_logs/kraken_bot_performance.log'))}

    results = {}
    for name, path in log_files.items():
        exists = os.path.exists(path)
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            results[name] = {
                'exists': True,
                'size_mb': size_mb,
                'last_modified': datetime.fromtimestamp(
                    os.path.getmtime(path)).isoformat()}
        else:
            results[name] = {
                'exists': False,
                'size_mb': 0,
                'last_modified': None
            }

    return results


def analyze_log_content(config: Dict[str, Any]) -> Dict[str, int]:
    """Analyse le contenu des logs pour les erreurs et avertissements."""
    log_path = os.path.expanduser(
        config.get(
            'LOG_FILE',
            '~/kraken_bot_logs/kraken_bot.log'))
    if not os.path.exists(log_path):
        return {'errors': 0, 'warnings': 0}

    errors = 0
    warnings = 0

    with open(log_path, 'r') as f:
        for line in f:
            if 'ERROR' in line:
                errors += 1
            elif 'WARNING' in line:
                warnings += 1

    return {'errors': errors, 'warnings': warnings}
