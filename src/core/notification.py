"""
Module de notification pour le bot de trading Kraken.
Gère les notifications par email, Telegram et autres canaux.
"""

from typing import Dict, Optional, List
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = logging.getLogger(__name__)


class NotificationManager:
    """Gestionnaire de notifications pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de notifications."""
        self.config = config
        self.notification_channels = config.get('notification_channels', [])

    def send_email(self, subject: str, message: str) -> bool:
        """
        Envoie une notification par email.

        Args:
            subject: Sujet de l'email
            message: Corps du message

        Returns:
            True si l'email a été envoyé avec succès, False sinon
        """
        try:
            if 'email' not in self.notification_channels:
                return False

            email_config = self.config.get('email', {})

            msg = MIMEMultipart()
            msg['From'] = email_config.get('from')
            msg['To'] = email_config.get('to')
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP(email_config.get('smtp_server'),
                              email_config.get('smtp_port')) as server:
                server.starttls()
                server.login(email_config.get('username'),
                             email_config.get('password'))
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'envoi d'email: {str(e)}")
            return False

    def send_telegram(self, message: str) -> bool:
        """
        Envoie une notification Telegram.

        Args:
            message: Message à envoyer

        Returns:
            True si le message a été envoyé avec succès, False sinon
        """
        try:
            if 'telegram' not in self.notification_channels:
                return False

            telegram_config = self.config.get('telegram', {})

            url = f"https://api.telegram.org/bot{telegram_config.get('bot_token')}/sendMessage"
            data = {
                'chat_id': telegram_config.get('chat_id'),
                'text': message
            }

            response = requests.post(url, json=data)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Erreur lors de l'envoi Telegram: {str(e)}")
            return False

    def send_notification(self,
                          message: str,
                          subject: Optional[str] = None,
                          channels: Optional[List[str]] = None) -> Dict[str,
                                                                        bool]:
        """
        Envoie une notification via les canaux spécifiés.

        Args:
            message: Message à envoyer
            subject: Sujet (pour email)
            channels: Canaux de notification (email, telegram, etc.)

        Returns:
            Dictionnaire avec le statut d'envoi pour chaque canal
        """
        results = {}

        if channels is None:
            channels = self.notification_channels

        if 'email' in channels:
            results['email'] = self.send_email(
                subject or "Notification Kraken Bot", message)

        if 'telegram' in channels:
            results['telegram'] = self.send_telegram(message)

        return results
