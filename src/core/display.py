"""
Module d'affichage avancé pour le shell avec graphiques ASCII, journalisation, tableau de bord interactif et thèmes.
"""
import os
import sys
import time
import json
import logging
import logging.handlers
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import namedtuple
from pathlib import Path

# Types personnalisés
OHLC = namedtuple('OHLC', ['open', 'high', 'low', 'close'])
Candle = namedtuple(
    'Candle', [
        'timestamp', 'open', 'high', 'low', 'close', 'volume'])


class Theme(Enum):
    """Thèmes visuels disponibles."""
    DEFAULT = 'default'
    DARK = 'dark'
    LIGHT = 'light'
    MONOKAI = 'monokai'
    SOLARIZED = 'solarized'


class MessageType(Enum):
    """Types de messages pour le système de journalisation."""
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    DEBUG = auto()
    TRADE = auto()
    SIGNAL = auto()
    SYSTEM = auto()


@dataclass
class LogMessage:
    """Classe pour représenter un message de journal."""
    message_type: MessageType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le message en dictionnaire pour la sérialisation."""
        return {
            'type': self.message_type.name,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'content': self.content,
            'details': self.details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogMessage':
        """Crée un LogMessage à partir d'un dictionnaire."""
        return cls(
            message_type=MessageType[data['type']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data.get('source', 'system'),
            content=data['content'],
            details=data.get('details', {})
        )


class ThemeConfig:
    """Configuration des thèmes visuels."""

    THEMES = {
        Theme.DEFAULT: {
            'background': '\033[49m',
            'text': '\033[39m',
            'success': '\033[92m',
            'warning': '\033[93m',
            'error': '\033[91m',
            'info': '\033[96m',
            'debug': '\033[90m',
            'highlight': '\033[95m',
            'primary': '\033[94m',
            'secondary': '\033[36m',
            'border': '\033[90m',
            'reset': '\033[0m',
            'chart_up': '\033[92m█\033[0m',
            'chart_down': '\033[91m█\033[0m',
            'chart_neutral': '\033[90m█\033[0m',
        },
        Theme.DARK: {
            'background': '\033[40m',
            'text': '\033[37m',
            'success': '\033[32m',
            'warning': '\033[33m',
            'error': '\033[31m',
            'info': '\033[36m',
            'debug': '\033[90m',
            'highlight': '\033[35m',
            'primary': '\033[34m',
            'secondary': '\033[36m',
            'border': '\033[90m',
            'reset': '\033[0m',
        },
        # Ajoutez d'autres thèmes selon les besoins
    }

    @classmethod
    def get_theme(cls, theme: Theme = Theme.DEFAULT) -> Dict[str, str]:
        """Récupère la configuration d'un thème."""
        return cls.THEMES.get(theme, cls.THEMES[Theme.DEFAULT])


class LoggerManager:
    """Gestionnaire de journaux avancé."""

    def __init__(
            self,
            log_dir: str = "logs",
            max_bytes: int = 10 *
            1024 *
            1024,
            backup_count: int = 5):
        """Initialise le gestionnaire de journaux."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.loggers = {}
        self.log_queue = queue.Queue()
        self.running = False
        self.writer_thread = None

    def start(self):
        """Démarre le thread d'écriture des journaux."""
        if not self.running:
            self.running = True
            self.writer_thread = threading.Thread(
                target=self._write_logs, daemon=True)
            self.writer_thread.start()

    def stop(self):
        """Arrête le thread d'écriture des journaux."""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)

    def get_logger(self, name: str) -> logging.Logger:
        """Récupère un logger configuré."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)

            # Formateur
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Handler console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            # Handler fichier
            log_file = self.log_dir / f"{name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            # Ajout des handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

            self.loggers[name] = logger

        return self.loggers[name]

    def log_message(self, message: LogMessage):
        """Ajoute un message à la file d'attente des journaux."""
        self.log_queue.put(message)

    def _write_logs(self):
        """Écrit les journaux en arrière-plan."""
        while self.running or not self.log_queue.empty():
            try:
                message = self.log_queue.get(timeout=1.0)
                logger = self.get_logger(message.source)
                log_method = getattr(
                    logger, message.message_type.name.lower(), logger.info)
                log_message = f"{message.content}"
                if message.details:
                    log_message += f"\nDétails: {json.dumps(message.details, indent=2, ensure_ascii=False)}"
                log_method(log_message)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(
                    f"Erreur lors de l'écriture du journal: {e}",
                    file=sys.stderr)


class ChartRenderer:
    """Classe pour le rendu de graphiques ASCII."""

    @staticmethod
    def render_candlestick(
            data: List[Candle],
            width: int = 80,
            height: int = 20) -> str:
        """Affiche un graphique en chandeliers ASCII."""
        if not data:
            return "Aucune donnée à afficher"

        # Préparation des données
        prices = [c.close for c in data[-width:]]
        volumes = [c.volume for c in data[-width:]]
        timestamps = [datetime.fromtimestamp(
            c.timestamp).strftime('%H:%M') for c in data[-width:]]

        # Calcul des bornes
        min_price = min(c.low for c in data[-width:])
        max_price = max(c.high for c in data[-width:])
        price_range = max(0.0001, max_price - min_price)

        # Création du graphique
        chart = []

        # Corps du graphique
        for y in range(height, 0, -1):
            line = []
            price_level = min_price + (price_range * (y - 1) / height)

            for i, candle in enumerate(data[-width:]):
                # Déterminer le caractère à afficher
                if candle.low <= price_level <= candle.high:
                    if candle.open < candle.close:  # Bougie haussière
                        char = '█' if candle.open <= price_level <= candle.close else '│'
                        line.append(f"\033[92m{char}\033[0m")  # Vert
                    elif candle.open > candle.close:  # Bougie baissière
                        char = '█' if candle.close <= price_level <= candle.open else '│'
                        line.append(f"\033[91m{char}\033[0m")  # Rouge
                    else:  # Ouverture = Clôture
                        line.append('─')
                else:
                    line.append(' ')

            # Ajout de l'échelle des prix
            price_str = f"{price_level:.8f}".rjust(12)
            chart.append(f"{price_str} │ {''.join(line)}")

        # Ligne du bas avec les volumes
        if volumes:
            max_volume = max(volumes) or 1
            volume_scale = 10.0 / max_volume
            volume_line = [' ' * 15]  # Espace pour l'échelle

            for vol in volumes[-width:]:
                bar_height = min(int(vol * volume_scale), 10)
                volume_line.append(
                    '▁▂▃▄▅▆▇█'[
                        bar_height -
                        1] if bar_height > 0 else ' ')

            chart.append(''.join(volume_line))

        # Légende du temps
        if timestamps:
            time_marks = [' ' * 15]  # Espace pour l'échelle
            step = max(1, len(timestamps) // 5)

            for i in range(0, len(timestamps), step):
                if i < len(time_marks[0]):
                    time_marks[0] = (
                        time_marks[0][:i] +
                        timestamps[i] +
                        time_marks[0][i + len(timestamps[i]):]
                    )

            chart.append(''.join(time_marks))

        return '\n'.join(chart)

    @staticmethod
    def render_indicator(
            values: List[float],
            width: int = 80,
            height: int = 10,
            upper_bound: float = None,
            lower_bound: float = None,
            overbought: float = None,
            oversold: float = None) -> str:
        """Affiche un indicateur technique sous forme de graphique ASCII."""
        if not values:
            return "Aucune donnée à afficher"

        # Ajuster la largeur au nombre de valeurs disponibles
        width = min(width, len(values))
        values = values[-width:]

        # Calculer les bornes si non spécifiées
        if upper_bound is None:
            upper_bound = max(values) if values else 1.0
        if lower_bound is None:
            lower_bound = min(values) if values else 0.0

        value_range = max(0.0001, upper_bound - lower_bound)

        # Créer le graphique
        chart = []

        for y in range(height, 0, -1):
            level = lower_bound + (value_range * (y - 1) / height)
            line = []

            for val in values:
                if val >= level:
                    line.append('█')
                else:
                    line.append(' ')

            # Ajouter des lignes de référence
            if overbought is not None and level <= overbought + \
                    (value_range * 0.02) and level >= overbought - (value_range * 0.02):
                line_str = ' '.join(line)
                # Rouge pour la survente
                line_str = f"\033[91m{line_str}\033[0m"
            elif oversold is not None and level <= oversold + (value_range * 0.02) and level >= oversold - (value_range * 0.02):
                line_str = ' '.join(line)
                # Vert pour la survente
                line_str = f"\033[92m{line_str}\033[0m"
            else:
                line_str = ' '.join(line)

            chart.append(f"{level:6.2f} | {line_str}")

        return '\n'.join(chart)


class InteractiveDashboard:
    """Tableau de bord interactif pour affichage en temps réel."""

    def __init__(
            self,
            title: str = "Tableau de bord",
            refresh_rate: float = 1.0):
        """Initialise le tableau de bord."""
        self.title = title
        self.refresh_rate = refresh_rate
        self.widgets = []
        self.running = False
        self.refresh_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def add_widget(
            self,
            name: str,
            update_func: Callable[[],
                                  str],
            interval: float = 1.0):
        """Ajoute un widget au tableau de bord."""
        self.widgets.append({
            'name': name,
            'update_func': update_func,
            'interval': interval,
            'last_update': 0,
            'content': 'Chargement...'
        })

    def start(self):
        """Démarre le rafraîchissement du tableau de bord."""
        if not self.running:
            self.running = True
            self.refresh_thread = threading.Thread(
                target=self._refresh_loop, daemon=True)
            self.refresh_thread.start()

    def stop(self):
        """Arrête le rafraîchissement du tableau de bord."""
        self.running = False
        self.stop_event.set()
        if self.refresh_thread:
            self.refresh_thread.join(timeout=2.0)

    def _refresh_loop(self):
        """Boucle de rafraîchissement des widgets."""
        while self.running and not self.stop_event.is_set():
            try:
                self._update_widgets()
                self._render()
                time.sleep(self.refresh_rate)
            except Exception as e:
                print(f"Erreur lors du rafraîchissement: {e}")

    def _update_widgets(self):
        """Met à jour le contenu des widgets."""
        current_time = time.time()

        with self.lock:
            for widget in self.widgets:
                if current_time - widget['last_update'] >= widget['interval']:
                    try:
                        widget['content'] = widget['update_func']()
                        widget['last_update'] = current_time
                    except Exception as e:
                        widget['content'] = f"Erreur: {str(e)}"

    def _render(self):
        """Affiche le tableau de bord."""
        # Effacer l'écran et positionner le curseur en haut à gauche
        print("\033[H\033[J", end="")

        # Afficher le titre
        print(f"\n{' ' + self.title + ' ':-^80}\n")

        # Afficher les widgets
        with self.lock:
            for i, widget in enumerate(self.widgets):
                print(f"=== {widget['name']} ===")
                print(widget['content'])
                if i < len(self.widgets) - 1:
                    print()  # Espace entre les widgets

        # Afficher les commandes
        print("\n" + "=" * 80)
        print("Commandes: [Q] Quitter  [R] Rafraîchir  [H] Aide")

    def handle_input(self, key: str) -> bool:
        """Gère les entrées utilisateur."""
        key = key.upper()

        if key == 'Q':
            self.stop()
            return False
        elif key == 'R':
            self._update_widgets()
            self._render()
        elif key == 'H':
            self._show_help()

        return True

    def _show_help(self):
        """Affiche l'aide."""
        help_text = """
        Aide du tableau de bord:

        Commandes disponibles:
        - Q: Quitter le tableau de bord
        - R: Forcer le rafraîchissement
        - H: Afficher cette aide

        Navigation:
        - Flèches: Naviguer entre les widgets
        - Entrée: Sélectionner un widget
        - Échap: Retour
        """

        print("\033[H\033[J", end="")  # Effacer l'écran
        print(help_text)
        input("\nAppuyez sur Entrée pour continuer...")
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ConsoleDisplay:
    """Classe principale pour l'affichage dans la console avec des fonctionnalités avancées."""

    def __init__(self, theme: Theme = Theme.DEFAULT, log_dir: str = "logs"):
        """Initialise l'affichage avec un thème et un répertoire de logs."""
        self.theme = ThemeConfig.get_theme(theme)
        self.logger = LoggerManager(log_dir=log_dir)
        self.logger.start()
        self.dashboard = None
        self._spinner_thread = None
        self._spinner_running = False

        # Configuration des couleurs
        self.COLORS = {
            'HEADER': '\033[95m',  # Magenta vif
            'ENDC': '\033[0m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m',
            'GRAY': '\033[90m',
            'RED': '\033[91m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'WHITE': '\033[97m',
            'BRIGHT_RED': '\033[91;1m',
            'BRIGHT_GREEN': '\033[92;1m',
            'BRIGHT_YELLOW': '\033[93;1m',
            'BRIGHT_BLUE': '\033[94;1m',
            'BRIGHT_MAGENTA': '\033[95;1m',
            'BRIGHT_CYAN': '\033[96;1m',
            'BRIGHT_WHITE': '\033[97;1m',
        }

        # Icônes et symboles
        self.ICONS = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'rocket': '🚀',
            'chart': '📊',
            'money': '💰',
            'clock': '⏱️',
            'search': '🔍',
            'gear': '⚙️',
            'brain': '🧠',
            'check': '✓',
            'cross': '✗',
            'arrow_up': '↑',
            'arrow_down': '↓',
            'neutral': '•',
            'buy': '🔼',
            'sell': '🔽',
            'hold': '⏸️'
        }

        # Configuration du thème
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme):
        """Applique un thème visuel."""
        self.theme = ThemeConfig.get_theme(theme)

    def log(self, message: str, message_type: MessageType = MessageType.INFO,
            details: Dict[str, Any] = None, source: str = None):
        """Enregistre un message dans les journaux."""
        log_msg = LogMessage(
            message_type=message_type,
            content=message,
            details=details or {},
            source=source or "console"
        )
        self.logger.log_message(log_msg)

    # === Méthodes d'affichage de base ===

    def clear_screen(self):
        """Efface l'écran de la console."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str, width: int = 80, color: str = 'HEADER'):
        """Affiche un en-tête stylisé."""
        self.clear_screen()
        border = self.theme.get('border', self.COLORS['GRAY'])
        reset = self.theme.get('reset', self.COLORS['ENDC'])
        print(f"\n{border}{'=' * width}{reset}")
        print(
            f"{self.theme.get(color.lower(), self.COLORS[color])}{title.upper():^{width}}{reset}")
        print(f"{border}{'=' * width}{reset}\n")

    def print_section(self, title: str, level: int = 1, prefix: str = ""):
        """Affiche une section avec un titre."""
        if level == 1:
            color = self.theme.get('primary', self.COLORS['BLUE'])
            if not prefix:
                prefix = '#'
        elif level == 2:
            color = self.theme.get('secondary', self.COLORS['CYAN'])
            if not prefix:
                prefix = '##'
        else:
            color = self.theme.get('text', self.COLORS['ENDC'])
            if not prefix:
                prefix = '###'

        reset = self.theme.get('reset', self.COLORS['ENDC'])
        print(f"\n{color}{prefix} {title}{reset}")

    def print_status(
            self,
            message: str,
            status: str = 'info',
            icon: Optional[str] = None,
            end: str = '\n',
            flush: bool = False):
        """Affiche un message de statut avec une icône et une couleur appropriée."""
        status_colors = {
            'success': self.theme.get('success', self.COLORS['GREEN']),
            'error': self.theme.get('error', self.COLORS['RED']),
            'warning': self.theme.get('warning', self.COLORS['YELLOW']),
            'info': self.theme.get('info', self.COLORS['CYAN']),
            'debug': self.theme.get('debug', self.COLORS['GRAY'])
        }

        color = status_colors.get(
            status.lower(), self.theme.get(
                'text', self.COLORS['ENDC']))
        icon = icon or self.ICONS.get(status.lower(), '•')
        reset = self.theme.get('reset', self.COLORS['ENDC'])

        print(f"{color}{icon} {message}{reset}", end=end, flush=flush)

        # Journalisation automatique
        log_type = getattr(MessageType, status.upper(), MessageType.INFO)
        self.log(message, log_type)

    # === Graphiques et visualisation ===

    def render_candlestick(
            self, data: List[Dict[str, float]], width: int = 80, height: int = 20) -> str:
        """Affiche un graphique en chandeliers ASCII."""
        if not data:
            return "Aucune donnée à afficher"

        # Convertir les données au format Candle
        candles = []
        for d in data:
            candle = Candle(
                timestamp=d.get('timestamp', 0),
                open=d.get('open', 0),
                high=d.get('high', 0),
                low=d.get('low', 0),
                close=d.get('close', 0),
                volume=d.get('volume', 0)
            )
            candles.append(candle)

        return ChartRenderer.render_candlestick(candles, width, height)

    def render_indicator(self, values: List[float], **kwargs) -> str:
        """Affiche un indicateur technique sous forme de graphique ASCII."""
        return ChartRenderer.render_indicator(values, **kwargs)

    # === Tableau de bord interactif ===

    def create_dashboard(
            self,
            title: str = "Tableau de bord",
            refresh_rate: float = 1.0):
        """Crée un nouveau tableau de bord interactif."""
        self.dashboard = InteractiveDashboard(title, refresh_rate)
        return self.dashboard

    def add_dashboard_widget(
            self,
            name: str,
            update_func: Callable[[],
                                  str],
            interval: float = 1.0):
        """Ajoute un widget au tableau de bord actif."""
        if self.dashboard is None:
            self.create_dashboard()
        self.dashboard.add_widget(name, update_func, interval)

    def start_dashboard(self):
        """Démarre le tableau de bord interactif."""
        if self.dashboard is not None:
            self.dashboard.start()

            # Boucle principale du tableau de bord
            try:
                import tty
                import termios
                import sys

                # Sauvegarder les paramètres du terminal
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)

                try:
                    tty.setcbreak(sys.stdin.fileno())

                    while True:
                        if sys.stdin in select.select(
                                [sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1)
                            if not self.dashboard.handle_input(key):
                                break

                finally:
                    # Restaurer les paramètres du terminal
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    self.dashboard.stop()

            except ImportError:
                # Mode non interactif (pour les tests)
                print(
                    "Mode interactif non disponible. Utilisation du mode automatique...")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.dashboard.stop()

    # === Animations et chargement ===

    def show_loading(
            self,
            message: str = "Chargement",
            done_message: str = "Terminé!",
            animation_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
            delay: float = 0.1):
        """Affiche une animation de chargement."""
        self._spinner_running = True

        def animate():
            i = 0
            while self._spinner_running:
                sys.stdout.write(
                    f"\r{message} {animation_chars[i % len(animation_chars)]} ")
                sys.stdout.flush()
                time.sleep(delay)
                i += 1

            # Effacer la ligne d'animation
            sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
            if done_message:
                self.print_status(done_message, 'success')
            else:
                sys.stdout.flush()

        self._spinner_thread = threading.Thread(target=animate, daemon=True)
        self._spinner_thread.start()

    def stop_loading(self):
        """Arrête l'animation de chargement."""
        self._spinner_running = False
        if self._spinner_thread:
            self._spinner_thread.join(timeout=1.0)

    def progress_bar(
            self,
            iteration: int,
            total: int,
            prefix: str = '',
            suffix: str = '',
            length: int = 50,
            fill: str = '█',
            show_percent: bool = True,
            show_count: bool = True,
            colorize: bool = True):
        """Affiche une barre de progression avancée."""
        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        # Utiliser les couleurs du thème
        if colorize:
            if iteration == total:
                color = self.theme.get('success', self.COLORS['GREEN'])
            elif percent > 70:
                color = self.theme.get('success', self.COLORS['GREEN'])
            elif percent > 40:
                color = self.theme.get('warning', self.COLORS['YELLOW'])
            else:
                color = self.theme.get('error', self.COLORS['RED'])
        else:
            color = ''

        # Construction de la ligne de progression
        progress_parts = []
        if prefix:
            progress_parts.append(prefix)

        progress_parts.append(
            f"|{color}{bar}{self.theme.get('reset', self.COLORS['ENDC'])}|")

        if show_percent:
            progress_parts.append(f" {percent:.1f}%")

        if show_count:
            progress_parts.append(f" ({iteration}/{total})")

        if suffix:
            progress_parts.append(f" {suffix}")

        # Afficher la barre de progression
        print('\r' + ' '.join(progress_parts), end='', flush=True)

        # Retour à la ligne lorsque terminé
        if iteration == total:
            print()

    # === Affichage de données structurées ===

    def print_table(self,
                    data: List[Dict[str,
                                    Any]],
                    headers: Dict[str,
                                  str] = None,
                    title: str = None,
                    align: Dict[str,
                                str] = None,
                    show_total: bool = False):
        """Affiche des données sous forme de tableau formaté avec option de total."""
        if not data:
            return

        # Déterminer les colonnes à afficher
        if headers is None:
            headers = {key: key.capitalize() for key in data[0].keys()}

        columns = list(headers.keys())

        # Calculer les largeurs de colonnes
        col_widths = {}
        for col in columns:
            # Largeur de l'en-tête
            header_width = len(str(headers.get(col, col)))
            # Largeur maximale des données
            data_width = max(len(str(row.get(col, ''))) for row in data)
            # Prendre le maximum entre les deux
            col_widths[col] = max(header_width, data_width, len(str(col)))

        # Afficher le titre si spécifié
        if title:
            total_width = sum(col_widths.values()) + 3 * (len(columns) - 1) + 4
            print(
                f"\n{self.theme.get('highlight', self.COLORS['BOLD'])}{title.upper():^{total_width}}{self.theme.get('reset', self.COLORS['ENDC'])}")

        # Afficher les en-têtes
        header_parts = []
        for col in columns:
            header_text = str(headers.get(col, col))
            width = col_widths[col]
            header_parts.append(
                f"{self.theme.get('bold', self.COLORS['BOLD'])}{header_text:<{width}}{self.theme.get('reset', self.COLORS['ENDC'])}")

        print("  ".join(header_parts))
        print(self.theme.get('border', self.COLORS['GRAY']) + "-" * (sum(col_widths.values(
        )) + 3 * (len(columns) - 1)) + self.theme.get('reset', self.COLORS['ENDC']))

        # Afficher les données
        for row in data:
            row_parts = []
            for col in columns:
                value = row.get(col, '')
                width = col_widths[col]
                align_char = '<'  # Par défaut à gauche

                # Vérifier l'alignement personnalisé
                if align and col in align:
                    if align[col].lower() == 'right':
                        align_char = '>'
                    elif align[col].lower() == 'center':
                        align_char = '^'

                # Aligner les nombres à droite par défaut
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    align_char = '>'

                row_parts.append(f"{value:{align_char}{width}}")

            print("  ".join(row_parts))

        # Afficher le total si demandé
        if show_total:
            total_parts = []
            for col in columns:
                total_value = sum(float(row.get(col, 0)) for row in data)
                width = col_widths[col]
                total_parts.append(f"{total_value:>{width}.2f}")

            print("  ".join(total_parts))

    def print_status_alt(self,
                         message: str,
                         status: str = 'info',
                         icon: Optional[str] = None,
                         end: str = '\n',
                         flush: bool = False,
                         details: Dict[str,
                                       Any] = None):
        """
        Affiche un message de statut avec une icône et une couleur appropriée.

        Args:
            message: Le message à afficher
            status: Type de statut ('success', 'error', 'warning', 'info')
            icon: Icône personnalisée (optionnel)
            end: Caractère de fin de ligne
            flush: Si True, force le vidage du buffer de sortie
            details: Détails supplémentaires à afficher (optionnel)
        """
        status_colors = {
            'success': 'BRIGHT_GREEN',
            'error': 'BRIGHT_RED',
            'warning': 'BRIGHT_YELLOW',
            'info': 'BRIGHT_CYAN',
            'debug': 'BRIGHT_MAGENTA'
        }

        color = status_colors.get(status.lower(), 'CYAN')
        icon = icon or self.ICONS.get(status.lower(), '•')

        print(f"{self.COLORS[color]}{icon} {message}{self.COLORS['ENDC']}",
              end=end, flush=flush)

    def print_trade_signal(
        self,
        pair: str,
        signal: str,
        price: float,
        indicators: Dict[str, Any],
        confidence: float = 1.0,
        details: Dict[str, Any] = None
    ):
        """Affiche un signal de trading détaillé."""
        signal_color = 'BRIGHT_GREEN' if signal.upper(
        ) == 'BUY' else 'BRIGHT_RED' if signal.upper() == 'SELL' else 'BRIGHT_YELLOW'
        confidence_pct = f"{confidence * 100:.1f}%"

        print(
            f"\n{self.theme.get('highlight', self.COLORS['BOLD'])}{'=' * 80}")
        print(
            f"{self.ICONS[signal.lower()] if signal.lower() in self.ICONS else '•'} "
            f"SIGNAL DE {signal.upper()} - {pair}")
        print(f"{'=' * 80}")

        # Informations principales
        print(
            f"{self.theme.get('info', self.COLORS['BRIGHT_CYAN'])}Informations principales:{self.theme.get('reset', self.COLORS['ENDC'])}")
        print(
            f"- Prix actuel: {self.theme.get('highlight', self.COLORS[signal_color])}{price:.8f} {pair.split('/')[1] if '/' in pair else 'USD'}{self.theme.get('reset', self.COLORS['ENDC'])}")
        print(
            f"- Confiance: {self.theme.get('highlight', self.COLORS[signal_color])}{confidence_pct}{self.theme.get('reset', self.COLORS['ENDC'])}")
        print(f"- Heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Afficher les indicateurs
        if indicators:
            print(
                f"\n{self.theme.get('info', self.COLORS['BRIGHT_CYAN'])}Indicateurs techniques:{self.theme.get('reset', self.COLORS['ENDC'])}")
            for name, value in indicators.items():
                if isinstance(value, (int, float)):
                    print(f"- {name}: {value:.4f}")
                else:
                    print(f"- {name}: {value}")

        # Afficher les détails supplémentaires
        if details:
            print(
                f"\n{self.theme.get('info', self.COLORS['BRIGHT_CYAN'])}Détails supplémentaires:{self.theme.get('reset', self.COLORS['ENDC'])}")
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    print(f"- {key}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"- {key}:")
                    for subkey, subvalue in value.items():
                        print(f"  - {subkey}: {subvalue}")
                else:
                    print(f"- {key}: {value}")

        print(
            f"{self.theme.get('highlight', self.COLORS['BOLD'])}{'=' * 80}{self.theme.get('reset', self.COLORS['ENDC'])}\n")

    def print_portfolio_summary(
        self,
        balance: Dict[str, float],
        current_prices: Dict[str, float],
        initial_balance: float = 1000.0
    ):
        """Affiche un résumé du portefeuille."""
        total_value = 0.0
        base_currency = 'USD'  # Devise de référence

        print(f"\n{self.COLORS['BOLD']}{'=' * 60}")
        print(f"📊 RÉSUMÉ DU PORTEFEUILLE")
        print(f"{'=' * 60}{self.COLORS['ENDC']}")

        # Afficher les soldes
        print(f"\n{self.COLORS['BOLD']}SOLDE DISPONIBLE:{self.COLORS['ENDC']}")
        for asset, amount in balance.items():
            if amount > 0.0001:  # Ne pas afficher les soldes négligeables
                value = amount * \
                    current_prices.get(asset, 0) if asset != base_currency else amount
                total_value += value
                print(
                    f"- {asset}: {self.COLORS['GREEN']}{amount:.8f}{self.COLORS['ENDC']} "
                    f"({value:.2f} {base_currency})")

        # Calculer la performance
        if initial_balance > 0:
            pnl = total_value - initial_balance
            pnl_pct = (pnl / initial_balance) * 100
            pnl_color = 'GREEN' if pnl >= 0 else 'RED'

            print(f"\n{self.COLORS['BOLD']}PERFORMANCE:{self.COLORS['ENDC']}")
            print(
                f"- Valeur totale: {self.COLORS['BOLD']}{total_value:.2f} {base_currency}{self.COLORS['ENDC']}")
            print(
                f"- P&L: {self.COLORS[pnl_color]}{'+' if pnl >= 0 else ''}{pnl:.2f} {base_currency} "
                f"({'+' if pnl >= 0 else ''}{pnl_pct:.2f}%){self.COLORS['ENDC']}")

        print(f"{self.COLORS['BOLD']}{'=' * 60}{self.COLORS['ENDC']}\n")

    def print_market_conditions(self, conditions: Dict[str, Any]):
        """Affiche les conditions actuelles du marché."""
        print(f"\n{self.COLORS['BOLD']}📈 ÉTAT DU MARCHÉ{self.COLORS['ENDC']}")
        print(f"{self.COLORS['GRAY']}{'-' * 40}{self.COLORS['ENDC']}")

        for key, value in conditions.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  - {k}: {v}")
            else:
                print(f"- {key}: {value}")

    def _get_colored_confidence(self, confidence: str) -> str:
        """Retourne le niveau de confiance avec un code couleur approprié."""
        try:
            value = float(confidence.strip('%'))
            if value >= 80:
                return f"{self.COLORS['GREEN']}{confidence}{self.COLORS['ENDC']}"
            elif value >= 60:
                return f"{self.COLORS['YELLOW']}{confidence}{self.COLORS['ENDC']}"
            else:
                return f"{self.COLORS['RED']}{confidence}{self.COLORS['ENDC']}"
        except (ValueError, AttributeError):
            return confidence

    def progress_bar_alt(
            self,
            iteration: int,
            total: int,
            prefix: str = '',
            suffix: str = '',
            length: int = 50,
            fill: str = '█',
            show_percent: bool = True,
            show_count: bool = True,
            colorize: bool = True):
        """Affiche une barre de progression avancée."""
        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        # Utiliser les couleurs du thème
        if colorize:
            if iteration == total:
                color = self.theme.get('success', self.COLORS['GREEN'])
            elif percent > 70:
                color = self.theme.get('success', self.COLORS['GREEN'])
            elif percent > 40:
                color = self.theme.get('warning', self.COLORS['YELLOW'])
            else:
                color = self.theme.get('error', self.COLORS['RED'])
        else:
            color = ''

        # Construction de la ligne de progression
        progress_parts = []
        if prefix:
            progress_parts.append(prefix)

        progress_parts.append(
            f"|{color}{bar}{self.theme.get('reset', self.COLORS['ENDC'])}|")

        if show_percent:
            progress_parts.append(f" {percent:.1f}%")

        if show_count:
            progress_parts.append(f" ({iteration}/{total})")

        if suffix:
            progress_parts.append(f" {suffix}")

        # Afficher la barre de progression
        print('\r' + ' '.join(progress_parts), end='', flush=True)

        # Retour à la ligne lorsque terminé
        if iteration == total:
            print()

    def loading_animation(
            self,
            message: str = "Chargement",
            done_message: str = "Terminé!",
            animation_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
            delay: float = 0.1):
        """Affiche une animation de chargement avec un message."""
        stop_event = threading.Event()

        def animate():
            i = 0
            while not stop_event.is_set():
                sys.stdout.write(
                    f"\r{message} {animation_chars[i % len(animation_chars)]} ")
                sys.stdout.flush()
                time.sleep(delay)
                i += 1

            # Effacer la ligne d'animation
            sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
            if done_message:
                self.print_status(done_message, 'success')
            else:
                sys.stdout.flush()

        thread = threading.Thread(target=animate)
        thread.daemon = True
        thread.start()

        def stop():
            stop_event.set()
            thread.join()

        return stop


# Instance par défaut pour une utilisation facile
display = ConsoleDisplay()
