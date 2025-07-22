import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from src.core.config_adapter import Config  # Migré vers le nouvel adaptateur
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from src.utils import helpers

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingReporter:
    """Génère des rapports de trading et envoie des notifications."""

    def __init__(self, config: Config):
        """Initialise le générateur de rapports."""
        self.config = config
        reporting_config = config.get_config('reporting')
        self.report_dir = os.path.expanduser(
            reporting_config.get('report_dir', '~/kraken_bot_reports'))
        
        # Récupérer la configuration email depuis la configuration de reporting
        self.email_config = {
            'enabled': reporting_config.get('email_enabled', False),
            'recipients': reporting_config.get('email_recipients', []),
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.example.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'smtp_username': os.getenv('SMTP_USERNAME', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('SMTP_FROM_EMAIL', 'noreply@example.com')
        }

        # Créer le répertoire des rapports s'il n'existe pas
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

    def generate_daily_report(self, trades: List[Dict[str, Any]]) -> str:
        """Génère un rapport quotidien des trades."""
        try:
            # Créer le DataFrame
            df = pd.DataFrame(trades)

            # Calculer les statistiques
            total_trades = len(df)
            win_rate = (df['profit'] > 0).mean() * 100
            total_profit = df['profit'].sum()
            avg_profit = df['profit'].mean()
            max_drawdown = df['profit'].min()

            # Créer le résumé
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades': total_trades,
                'win_rate': f"{win_rate:.1f}%",
                'total_profit': f"${total_profit:.2f}",
                'avg_profit': f"${avg_profit:.2f}",
                'max_drawdown': f"${max_drawdown:.2f}"
            }

            # Sauvegarder le rapport
            report_path = os.path.join(
                self.report_dir,
                f"daily_report_{datetime.now().strftime('%Y%m%d')}.json")
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=4)

            # Générer les graphiques si pandas est installé
            self._generate_charts(df)

            return report_path

        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            return None

    def _generate_charts(self, df: pd.DataFrame):
        """Génère des graphiques pour le rapport."""
        try:
            # Créer un graphique des profits
            plt.figure(figsize=(10, 6))
            plt.plot(df['profit'].cumsum(), marker='o')
            plt.title('Performance Cumulée')
            plt.xlabel('Trade')
            plt.ylabel('Profit Cumulé ($)')
            plt.grid(True)
            plt.savefig(os.path.join(self.report_dir, 'cumulative_profit.png'))

            # Créer un histogramme des profits
            plt.figure(figsize=(10, 6))
            plt.hist(df['profit'], bins=20)
            plt.title('Distribution des Profits')
            plt.xlabel('Profit ($)')
            plt.ylabel('Nombre de Trades')
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    self.report_dir,
                    'profit_distribution.png'))

        except Exception as e:
            logger.error(
                f"Erreur lors de la génération des graphiques: {str(e)}")

    def send_email_report(self, report_path: str):
        """Envoie un rapport par email."""
        try:
            if not self.email_config.get('enabled', False):
                return

            # Configurer l'email
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = f"Rapport de Trading - {datetime.now().strftime('%Y-%m-%d')}"

            # Lire le rapport
            with open(report_path, 'r') as f:
                report_data = json.load(f)

            # Créer le corps de l'email
            body = f"""
            Rapport de Trading - {report_data['date']}

            Statistiques:
            - Nombre total de trades: {report_data['total_trades']}
            - Taux de réussite: {report_data['win_rate']}
            - Profit total: {report_data['total_profit']}
            - Profit moyen: {report_data['avg_profit']}
            - Drawdown maximum: {report_data['max_drawdown']}
            """

            msg.attach(MIMEText(body))

            # Ajouter les graphiques
            for chart in ['cumulative_profit.png', 'profit_distribution.png']:
                chart_path = os.path.join(self.report_dir, chart)
                if os.path.exists(chart_path):
                    with open(chart_path, 'rb') as f:
                        part = MIMEApplication(f.read(), Name=chart)
                        part['Content-Disposition'] = f'attachment; filename="{chart}"'
                        msg.attach(part)

            # Envoyer l'email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config.get('smtp_port', 587)) as server:
                server.starttls()
                server.login(
                    self.email_config['smtp_user'],
                    self.email_config['smtp_password']
                )
                server.send_message(msg)

            logger.info("Rapport envoyé par email")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'envoi du rapport par email: {str(e)}")

    def generate_performance_metrics(
            self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcule les métriques de performance."""
        profits = [t['profit']
                   for t in trades if 'profit' in t and t['profit'] is not None]
        entry_prices = [t['entry_price']
                        for t in trades if 'entry_price' in t and t['entry_price'] is not None]
        returns = [
            p / e for p,
            e in zip(
                profits,
                entry_prices) if e != 0] if profits and entry_prices else []
        return {
            'sharpe_ratio': helpers.calculate_sharpe_ratio(returns) if returns else 0.0,
            'sortino_ratio': helpers.calculate_sortino_ratio(returns) if returns else 0.0,
            'max_drawdown': helpers.calculate_max_drawdown(returns) if returns else 0.0,
            'calmar_ratio': self._calculate_calmar_ratio(
                trades,
                returns)}

    def _calculate_calmar_ratio(
            self, trades: List[Dict[str, Any]], returns: list) -> float:
        try:
            annual_return = sum(
                [t['profit'] for t in trades if 'profit' in t]) / len(trades) if trades else 0.0
            max_dd = helpers.calculate_max_drawdown(
                returns) if returns else 0.0
            return annual_return / abs(max_dd) if max_dd != 0 else float('inf')
        except Exception:
            return 0.0


def setup_reporting(config: Dict[str, Any]) -> TradingReporter:
    """Configure le système de reporting."""
    reporter = TradingReporter(config)
    return reporter
