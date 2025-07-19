import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.status import Status
from src.core.trading import TradingManager
from src.core.logging import LoggerManager


class TradingShell:
    """Interface shell interactive pour le bot de trading."""

    def __init__(self, trader: TradingManager, refresh_rate: float = 1.0):
        self.trader = trader
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.logger = LoggerManager()
        self.status = Status("Initialisation du bot...")
        self.layout = self._create_layout()
        self.running = False
        self.task = None

    async def run(self):
        """Démarrer l'interface CLI."""
        with Live(self.layout, refresh_per_second=self.refresh_rate):
            self.running = True
            self.task = asyncio.create_task(self.update())
            try:
                while self.running:
                    await asyncio.sleep(self.refresh_rate)
            except asyncio.CancelledError:
                pass
            finally:
                self.running = False
                if self.task:
                    self.task.cancel()
                    try:
                        await self.task
                    except asyncio.CancelledError:
                        pass
            self.console.print("\n[bold red]Arrêt du bot...[/]")

    async def update(self):
        """Mettre à jour l'interface."""
        while self.running:
            try:
                # Vérifier la configuration avant chaque mise à jour
                if not await self.trader.is_config_valid():
                    self.console.print(
                        "[yellow]Configuration invalide. Vérifiez vos paramètres.[/yellow]")
                    continue

                # Mettre à jour les sections de l'interface
                await self._update_header()
                await self._update_main()
                await self._update_footer()

                await asyncio.sleep(self.refresh_rate)
            except Exception as e:
                error_msg = f"Erreur lors de la mise à jour: {str(e)}"
                self.logger.error(error_msg)
                self.console.print(f"[red]{error_msg}[/red]")

    def _create_layout(self) -> Layout:
        """Crée la mise en page de l'interface."""
        layout = Layout()

        # Créer les sections principales
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=7)
        )

        # Diviser le corps en deux colonnes
        layout['body'].split_row(
            Layout(name="stats", ratio=2),
            Layout(name="pairs", ratio=1)
        )

        return layout

    async def _update_header(self):
        """Mettre à jour l'en-tête."""
        header = Panel(
            Align.center(
                f"Bot de Trading Kraken - Mode Simulation",
                vertical="middle"
            ),
            style="bold white on blue"
        )
        self.layout['header'].update(header)

    async def _update_main(self):
        """Mettre à jour la section principale."""
        # Créer le tableau des statistiques
        stats_table = Table(title="Statistiques du Bot")
        stats_table.add_column("Statistique", style="cyan")
        stats_table.add_column("Valeur", style="green")

        # Ajouter les statistiques
        # Récupérer les valeurs async
        balance = await self.trader.get_account_balance('EUR')
        open_positions = len(await self.trader.get_open_positions())
        performance = await self.trader.get_account_performance()

        # Ajouter les statistiques
        stats_table.add_row("Solde EUR", f"{balance:,.2f}")
        stats_table.add_row("Nombre de trades", f"{open_positions}")
        stats_table.add_row("Performance", f"{performance:.2f}%")

        # Créer le tableau des paires de trading
        pairs_table = Table(title="Paires de Trading")
        pairs_table.add_column("Paire", style="cyan")
        pairs_table.add_column("Score", style="green")
        pairs_table.add_column("État", style="red")

        # Ajouter les paires de trading
        for pair, config in self.trader.pair_configs.items():
            score = await config.technical_analyzer.get_score(pair)
            status = "Actif" if pair in self.trader.active_pairs else "Inactif"
            pairs_table.add_row(
                pair,
                f"{score:.2f}",
                status
            )

        # Mettre à jour les sections
        self.layout['stats'].update(Panel(stats_table))
        self.layout['pairs'].update(Panel(pairs_table))

    async def _update_footer(self):
        """Mettre à jour le pied de page."""
        footer = Panel(
            Align.center(
                "Commandes: q - Quitter | h - Aide",
                vertical="middle"
            ),
            style="bold white on blue"
        )
        self.layout['footer'].update(footer)

    def _create_header(self) -> Panel:
        """Crée le panneau d'en-tête."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"[bold cyan]Kraken Trading Bot[/] - {current_time}"
        return Panel(header, style="bold white")

    def _create_market_summary(self) -> Table:
        """Crée le tableau de résumé du marché."""
        table = Table(title="[bold]Résumé du Marché[/]")
        table.add_column("Indicateur", style="cyan")
        table.add_column("Valeur", style="green")
        table.add_column("Tendance", style="yellow")

        # Récupérer les données du marché
        market_data = self.trader.market_trend_detector.get_market_metrics()

        for indicator, value in market_data.items():
            trend = "↑" if value >= 0 else "↓"
            table.add_row(
                indicator,
                f"{value:.2f}",
                trend
            )

        return table

    def _create_trading_summary(self) -> Table:
        """Crée le tableau de résumé du trading."""
        table = Table(title="[bold]Résumé du Trading[/]")
        table.add_column("Statistique", style="cyan")
        table.add_column("Valeur", style="green")

        # Récupérer les métriques globales
        metrics = self.trader.get_global_metrics()

        for metric, value in metrics.items():
            table.add_row(
                metric,
                f"{value:.2f}"
            )

        return table

    def _create_positions_table(self) -> Table:
        """Crée le tableau des positions actuelles."""
        table = Table(title="[bold]Positions Actuelles[/]")
        table.add_column("Paire", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Profit", style="yellow")
        table.add_column("Risque", style="red")

        # Récupérer les positions actuelles
        positions = self.trader.get_positions()

        for pair, position in positions.items():
            profit = position.get('profit', 0)
            risk = position.get('risk', 0)

            table.add_row(
                pair,
                f"{position.get('size', 0):.2f}",
                f"{profit:.2f} USD",
                f"{risk:.2%}"
            )

        return table

    def _create_commands_panel(self) -> Panel:
        """Crée le panneau des commandes."""
        commands = [
            "[bold]Commandes disponibles:[/]",
            "[cyan]h[/] - Aide",
            "[cyan]r[/] - Rotation des paires",
            "[cyan]s[/] - Statistiques détaillées",
            "[cyan]q[/] - Quitter"
        ]

        return Panel("\n".join(commands), title="[bold]Commandes[/]")

    def _update_layout(self):
        """Met à jour la mise en page."""
        # Mettre à jour l'en-tête
        self.layout['header'].update(self._create_header())

        # Mettre à jour les statistiques
        self.layout['stats'].update(self._create_trading_summary())

        # Mettre à jour les paires de trading
        self.layout['pairs'].update(self._create_positions_table())

        # Mettre à jour le pied de page
        self.layout['footer'].update(self._create_commands_panel())

    async def run(self):
        """Démarrer l'interface CLI."""
        with Live(self.layout, refresh_per_second=self.refresh_rate):
            self.running = True
            self.task = asyncio.create_task(self.update())
            try:
                while self.running:
                    await asyncio.sleep(self.refresh_rate)
            except asyncio.CancelledError:
                pass
            finally:
                self.running = False
                if self.task:
                    self.task.cancel()
                    try:
                        await self.task
                    except asyncio.CancelledError:
                        pass
            self.console.print("\n[bold red]Arrêt du bot...[/]")

    def _create_detailed_stats(self) -> Table:
        """Crée le tableau des statistiques détaillées."""
        table = Table(title="[bold]Statistiques Détaillées[/]")
        table.add_column("Paire", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Volatilité", style="yellow")
        table.add_column("Momentum", style="blue")
        table.add_column("Volume", style="magenta")

        # Récupérer les métriques détaillées
        detailed_metrics = self.trader.get_detailed_metrics()

        for pair, metrics in detailed_metrics.items():
            table.add_row(
                pair,
                f"{metrics.get('score', 0):.2f}",
                f"{metrics.get('volatility', 0):.2%}",
                f"{metrics.get('momentum', 0):.2%}",
                f"{metrics.get('volume', 0):.2f}"
            )

        return table


if __name__ == "__main__":
    import sys
    import asyncio
    from src.core.trading import TradingManager
    from src.core.api.kraken import KrakenAPI

    api = KrakenAPI()  # À adapter avec tes paramètres réels si besoin
    trader = TradingManager(api)
    shell = TradingShell(trader)
    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")
        sys.exit(0)
