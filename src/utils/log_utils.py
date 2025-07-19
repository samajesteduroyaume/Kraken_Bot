import json
import time
from typing import Any, Dict, List, Optional
from colorama import init
from datetime import datetime

# Initialiser colorama pour Windows
init(autoreset=True)


class LogUtils:
    @staticmethod
    def format_json(data: Any, indent: int = 2) -> str:
        """Formatte les données JSON pour une meilleure lisibilité."""
        try:
            return json.dumps(data, indent=indent, default=str)
        except (TypeError, ValueError):
            return str(data)

    @staticmethod
    def format_duration(start_time: float) -> str:
        """Formatte la durée depuis le temps de début."""
        duration = time.time() - start_time
        return f"{duration:.2f}s"

    @staticmethod
    def format_performance(metrics: Dict[str, Any]) -> str:
        """Formatte les métriques de performance."""
        return f"Performance: {LogUtils.format_json(metrics)}"

    @staticmethod
    def format_trade(trade: Dict[str, Any]) -> str:
        """Formatte les informations de trade."""
        return f"Trade: {LogUtils.format_json(trade)}"

    @staticmethod
    def format_position(position: Dict[str, Any]) -> str:
        """Formatte les informations de position."""
        return f"Position: {LogUtils.format_json(position)}"

    @staticmethod
    def format_balance(balance: Dict[str, Any]) -> str:
        """Formatte les informations de balance."""
        return f"Balance: {LogUtils.format_json(balance)}"

    @staticmethod
    def format_error(error: Exception, details: Optional[Dict] = None) -> str:
        """Formatte les erreurs avec des détails supplémentaires."""
        error_msg = str(error)
        if details:
            error_msg += f"\nDetails: {LogUtils.format_json(details)}"
        return error_msg

    @staticmethod
    def format_summary(stats: Dict[str, Any]) -> str:
        """Formatte un résumé des statistiques."""
        return f"Summary: {LogUtils.format_json(stats)}"

    @staticmethod
    def format_table(data: List[Dict], columns: List[str]) -> str:
        """Formatte les données en tableau."""
        if not data or not columns:
            return "No data available"

        # Créer les en-têtes
        headers = [f"{col.upper()}" for col in columns]

        # Créer les lignes
        rows = []
        for item in data:
            row = []
            for col in columns:
                value = item.get(col, "")
                if isinstance(value, (float, int)):
                    row.append(f"{value:,.2f}")
                else:
                    row.append(str(value))
            rows.append(row)

        # Calculer la largeur des colonnes
        widths = [max(len(str(row[i])) for row in [headers] + rows)
                  for i in range(len(columns))]

        # Créer la ligne de séparation
        sep = "|".join("-" * (width + 2) for width in widths)

        # Créer le tableau
        table = []
        table.append(sep)
        table.append(
            "| " +
            " | ".join(
                f"{headers[i]:<{widths[i]}}" for i in range(
                    len(columns))) +
            " |")
        table.append(sep)
        for row in rows:
            table.append(
                "| " +
                " | ".join(
                    f"{cell:<{widths[i]}}" for i,
                    cell in enumerate(row)) +
                " |")
        table.append(sep)

        return "\n".join(table)

    @staticmethod
    def format_status(status: Dict[str, Any]) -> str:
        """Formatte le statut du trading."""
        return f"Status: {LogUtils.format_json(status)}"

    @staticmethod
    def format_signal(signal: Dict[str, Any]) -> str:
        """Formatte les signaux de trading."""
        return f"Signal: {LogUtils.format_json(signal)}"

    @staticmethod
    def format_indicators(indicators: Dict[str, Any]) -> str:
        """Formatte les indicateurs techniques."""
        return f"Indicators: {LogUtils.format_json(indicators)}"

    @staticmethod
    def format_risk(risk: Dict[str, Any]) -> str:
        """Formatte les métriques de risque."""
        return f"Risk: {LogUtils.format_json(risk)}"

    @staticmethod
    def format_order(order: Dict[str, Any]) -> str:
        """Formatte les informations de l'ordre."""
        return f"Order: {LogUtils.format_json(order)}"

    @staticmethod
    def format_backtest(results: Dict[str, Any]) -> str:
        """Formatte les résultats de backtesting."""
        return f"Backtest Results: {LogUtils.format_json(results)}"

    @staticmethod
    def format_prediction(prediction: Dict[str, Any]) -> str:
        """Formatte les prédictions ML."""
        return f"Prediction: {LogUtils.format_json(prediction)}"

    @staticmethod
    def format_market_data(data: Dict[str, Any]) -> str:
        """Formatte les données de marché."""
        return f"Market Data: {LogUtils.format_json(data)}"

    @staticmethod
    def format_portfolio(portfolio: Dict[str, Any]) -> str:
        """Formatte le portefeuille."""
        return f"Portfolio: {LogUtils.format_json(portfolio)}"

    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """Formatte les métriques de performance."""
        return f"Metrics: {LogUtils.format_json(metrics)}"

    @staticmethod
    def format_config(config: Dict[str, Any]) -> str:
        """Formatte la configuration."""
        return f"Config: {LogUtils.format_json(config)}"

    @staticmethod
    def format_system_info(info: Dict[str, Any]) -> str:
        """Formatte les informations système."""
        return f"System Info: {LogUtils.format_json(info)}"

    @staticmethod
    def format_version(version: str) -> str:
        """Formatte la version."""
        return f"Version: {version}"

    @staticmethod
    def format_time(timestamp: float) -> str:
        """Formatte le timestamp en heure locale."""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def format_progress(current: int, total: int) -> str:
        """Formatte la progression."""
        percent = (current / total) * 100
        return f"Progress: {percent:.1f}% ({current}/{total})"

    @staticmethod
    def format_alert(message: str, level: str = "INFO") -> str:
        """Formatte les alertes."""
        return f"[{level}] {message}"

    @staticmethod
    def format_notification(message: str, type_: str = "INFO") -> str:
        """Formatte les notifications."""
        return f"[{type_}] {message}"

    @staticmethod
    def format_event(event: Dict[str, Any]) -> str:
        """Formatte les événements."""
        return f"Event: {LogUtils.format_json(event)}"

    @staticmethod
    def format_trade_summary(trades: List[Dict]) -> str:
        """Formatte un résumé des trades."""
        if not trades:
            return "No trades executed"

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades) * \
            100 if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in trades)

        return f"""
Trade Summary:
- Total Trades: {total_trades}
- Winning Trades: {winning_trades} ({win_rate:.1f}%)
- Total P&L: {total_pnl:.2f}
"""

    @staticmethod
    def format_performance_metrics(metrics: Dict[str, Any]) -> str:
        """Formatte les métriques de performance."""
        return f"""
Performance Metrics:
{LogUtils.format_json(metrics)}
"""

    @staticmethod
    def format_risk_metrics(metrics: Dict[str, Any]) -> str:
        """Formatte les métriques de risque."""
        return f"""
Risk Metrics:
{LogUtils.format_json(metrics)}
"""

    @staticmethod
    def format_market_analysis(analysis: Dict[str, Any]) -> str:
        """Formatte l'analyse du marché."""
        return f"""
Market Analysis:
{LogUtils.format_json(analysis)}
"""

    @staticmethod
    def format_strategy_summary(stats: Dict[str, Any]) -> str:
        """Formatte le résumé de la stratégie."""
        return f"""
Strategy Summary:
{LogUtils.format_json(stats)}
"""

    @staticmethod
    def format_backtest_summary(results: Dict[str, Any]) -> str:
        """Formatte le résumé du backtesting."""
        return f"""
Backtest Summary:
{LogUtils.format_json(results)}
"""
