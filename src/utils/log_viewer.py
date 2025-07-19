import json
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.live import Live
from rich.markdown import Markdown
from rich import box


class LogViewer:
    def __init__(self):
        self.console = Console()
        self.progress = Progress()

    def format_log(self, log: Dict) -> str:
        """Formatte un log JSON pour une meilleure lisibilité."""
        try:
            level = log.get('levelname', 'INFO')
            message = log.get('message', '')
            name = log.get('name', '')
            timestamp = log.get('asctime', '')

            # Créer une table pour le log
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("", style="cyan")
            table.add_column("", style="white")

            # Ajouter les informations du log
            table.add_row("Timestamp", timestamp)
            table.add_row("Level", level)
            table.add_row("Logger", name)
            table.add_row("Message", message)

            # Ajouter les informations supplémentaires
            if 'pair' in log:
                table.add_row("Pair", log['pair'])
            if 'type' in log:
                table.add_row("Type", log['type'])
            if 'duration' in log:
                table.add_row("Duration", f"{log['duration']:.2f}s")

            return str(table)

        except Exception as e:
            return f"Error formatting log: {str(e)}"

    def display_logs(self, logs: List[Dict]):
        """Affiche une liste de logs dans un tableau."""
        if not logs:
            self.console.print("[yellow]No logs available[/yellow]")
            return

        table = Table(title="Trading Bot Logs", box=box.DOUBLE_EDGE)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Level", style="magenta")
        table.add_column("Logger", style="green")
        table.add_column("Message", style="white")

        for log in logs:
            level = log.get('levelname', 'INFO')
            timestamp = log.get('asctime', '')
            name = log.get('name', '')
            message = log.get('message', '')

            table.add_row(
                timestamp,
                level,
                name,
                message
            )

        self.console.print(table)

    def display_summary(self, logs: List[Dict]):
        """Affiche un résumé des logs."""
        if not logs:
            self.console.print(
                "[yellow]No logs available for summary[/yellow]")
            return

        # Compter les logs par niveau
        level_counts = {}
        for log in logs:
            level = log.get('levelname', 'INFO')
            level_counts[level] = level_counts.get(level, 0) + 1

        # Créer un tableau pour le résumé
        summary = Table(title="Log Summary", box=box.DOUBLE_EDGE)
        summary.add_column("Level", style="magenta")
        summary.add_column("Count", style="cyan")

        for level, count in level_counts.items():
            summary.add_row(level, str(count))

        self.console.print(summary)

    def display_errors(self, logs: List[Dict]):
        """Affiche uniquement les erreurs."""
        errors = [log for log in logs if log.get(
            'levelname') in ['ERROR', 'CRITICAL']]
        if not errors:
            self.console.print("[green]No errors found![/green]")
            return

        table = Table(title="Error Logs", box=box.DOUBLE_EDGE)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Level", style="red")
        table.add_column("Logger", style="green")
        table.add_column("Message", style="white")

        for error in errors:
            table.add_row(
                error.get('asctime', ''),
                error.get('levelname', ''),
                error.get('name', ''),
                error.get('message', '')
            )

        self.console.print(table)

    def display_trades(self, logs: List[Dict]):
        """Affiche les logs de trades."""
        trades = [
            log for log in logs if 'trade' in log.get(
                'message', '').lower()]
        if not trades:
            self.console.print("[yellow]No trade logs found[/yellow]")
            return

        table = Table(title="Trade Logs", box=box.DOUBLE_EDGE)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Pair", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Amount", style="white")
        table.add_column("Price", style="white")

        for trade in trades:
            try:
                data = json.loads(trade.get('message', '{}'))
                table.add_row(
                    trade.get('asctime', ''),
                    data.get('pair', ''),
                    data.get('type', ''),
                    str(data.get('amount', '')),
                    str(data.get('price', ''))
                )
            except BaseException:
                continue

        self.console.print(table)

    def display_positions(self, logs: List[Dict]):
        """Affiche les logs de positions."""
        positions = [
            log for log in logs if 'position' in log.get(
                'message', '').lower()]
        if not positions:
            self.console.print("[yellow]No position logs found[/yellow]")
            return

        table = Table(title="Position Logs", box=box.DOUBLE_EDGE)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Pair", style="green")
        table.add_column("Side", style="magenta")
        table.add_column("Amount", style="white")
        table.add_column("Entry Price", style="white")

        for position in positions:
            try:
                data = json.loads(position.get('message', '{}'))
                table.add_row(
                    position.get('asctime', ''),
                    data.get('pair', ''),
                    data.get('side', ''),
                    str(data.get('amount', '')),
                    str(data.get('entry_price', ''))
                )
            except BaseException:
                continue

        self.console.print(table)

    def display_metrics(self, logs: List[Dict]):
        """Affiche les métriques de performance."""
        metrics = [
            log for log in logs if 'metrics' in log.get(
                'message', '').lower()]
        if not metrics:
            self.console.print("[yellow]No metrics logs found[/yellow]")
            return

        table = Table(title="Performance Metrics", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="cyan")

        for metric in metrics:
            try:
                data = json.loads(metric.get('message', '{}'))
                for key, value in data.items():
                    table.add_row(key, str(value))
            except BaseException:
                continue

        self.console.print(table)

    def display_live(self, logs: List[Dict]):
        """Affiche les logs en temps réel."""
        with Live(self._create_live_table(logs), refresh_per_second=4) as live:
            while True:
                try:
                    live.update(self._create_live_table(logs))
                except KeyboardInterrupt:
                    break

    def _create_live_table(self, logs: List[Dict]):
        """Crée une table pour l'affichage en temps réel."""
        table = Table(box=box.DOUBLE_EDGE)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Level", style="magenta")
        table.add_column("Logger", style="green")
        table.add_column("Message", style="white")

        for log in logs[-10:]:  # Afficher les 10 derniers logs
            table.add_row(
                log.get('asctime', ''),
                log.get('levelname', ''),
                log.get('name', ''),
                log.get('message', '')
            )

        return Panel(table, title="Live Logs")

    def display_markdown(self, content: str):
        """Affiche du contenu Markdown."""
        self.console.print(Markdown(content))

    def display_progress(self, total: int, current: int, message: str = ""):
        """Affiche une barre de progression."""
        progress = Progress()
        task = progress.add_task("Progress", total=total)

        with Live(progress):
            progress.update(task, completed=current, description=message)

    def display_alert(self, message: str, level: str = "INFO"):
        """Affiche une alerte."""
        style = {
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'SUCCESS': 'green'
        }.get(level.upper(), 'white')

        self.console.print(f"[{style}]{level}[/]: {message}")

    def display_notification(self, message: str, type_: str = "INFO"):
        """Affiche une notification."""
        style = {
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'SUCCESS': 'green'
        }.get(type_.upper(), 'white')

        self.console.print(f"[{style}]{type_}[/]: {message}")

    def display_event(self, event: Dict):
        """Affiche un événement."""
        try:
            table = Table(title="Event", box=box.DOUBLE_EDGE)
            table.add_column("Field", style="magenta")
            table.add_column("Value", style="cyan")

            for key, value in event.items():
                table.add_row(key, str(value))

            self.console.print(table)
        except Exception as e:
            self.console.print(f"[red]Error displaying event: {str(e)}[/]")

    def display_trade_summary(self, trades: List[Dict]):
        """Affiche un résumé des trades."""
        if not trades:
            self.console.print("[yellow]No trades to summarize[/yellow]")
            return

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades) * \
            100 if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in trades)

        table = Table(title="Trade Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="cyan")

        table.add_row("Total Trades", str(total_trades))
        table.add_row("Winning Trades", f"{winning_trades} ({win_rate:.1f}%)")
        table.add_row("Total P&L", f"{total_pnl:.2f}")

        self.console.print(table)

    def display_performance_metrics(self, metrics: Dict):
        """Affiche les métriques de performance."""
        if not metrics:
            self.console.print(
                "[yellow]No performance metrics available[/yellow]")
            return

        table = Table(title="Performance Metrics", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="cyan")

        for key, value in metrics.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def display_risk_metrics(self, metrics: Dict):
        """Affiche les métriques de risque."""
        if not metrics:
            self.console.print("[yellow]No risk metrics available[/yellow]")
            return

        table = Table(title="Risk Metrics", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="cyan")

        for key, value in metrics.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def display_market_analysis(self, analysis: Dict):
        """Affiche l'analyse du marché."""
        if not analysis:
            self.console.print("[yellow]No market analysis available[/yellow]")
            return

        table = Table(title="Market Analysis", box=box.DOUBLE_EDGE)
        table.add_column("Indicator", style="magenta")
        table.add_column("Value", style="cyan")

        for key, value in analysis.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def display_strategy_summary(self, stats: Dict):
        """Affiche le résumé de la stratégie."""
        if not stats:
            self.console.print("[yellow]No strategy stats available[/yellow]")
            return

        table = Table(title="Strategy Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="cyan")

        for key, value in stats.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def display_backtest_summary(self, results: Dict):
        """Affiche le résumé du backtesting."""
        if not results:
            self.console.print(
                "[yellow]No backtest results available[/yellow]")
            return

        table = Table(title="Backtest Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="cyan")

        for key, value in results.items():
            table.add_row(key, str(value))

        self.console.print(table)
