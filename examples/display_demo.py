#!/usr/bin/env python3
"""
Démonstration des fonctionnalités avancées du module d'affichage.
"""
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Ajouter le répertoire parent au chemin pour les imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.display import ConsoleDisplay, Theme, MessageType

def generate_sample_data(days: int = 30) -> List[Dict[str, float]]:
    """Génère des données de marché aléatoires pour la démo."""
    data = []
    base_price = 100.0
    timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
    
    for _ in range(days):
        open_price = base_price
        close_price = open_price * (1 + (random.random() - 0.5) * 0.1)  # +/- 5%
        high = max(open_price, close_price) * (1 + random.random() * 0.05)  # Jusqu'à 5% plus haut
        low = min(open_price, close_price) * (1 - random.random() * 0.05)  # Jusqu'à 5% plus bas
        volume = random.uniform(100, 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
        
        # Mise à jour pour la prochaine itération
        base_price = close_price
        timestamp += 86400  # +1 jour en secondes
    
    return data

def demo_basic_features(display: ConsoleDisplay):
    """Démonstration des fonctionnalités de base."""
    display.print_header("Démonstration des Fonctionnalités de Base")
    
    # Affichage de sections
    display.print_section("Messages de Statut", 1)
    display.print_status("Ceci est un message d'information", "info")
    display.print_status("Opération réussie!", "success")
    display.print_status("Attention: quelque chose d'inhabituel s'est produit", "warning")
    display.print_status("Une erreur critique est survenue", "error")
    
    # Barre de progression
    input("\nAppuyez sur Entrée pour voir une barre de progression...")
    display.clear_screen()
    display.print_section("Barre de Progression", 1)
    
    total = 100
    for i in range(total + 1):
        display.progress_bar(i, total, prefix="Progression:", suffix=f"Étape {i}/{total}")
        time.sleep(0.02)  # Simulation d'un traitement
    
    # Animation de chargement
    input("\n\nAppuyez sur Entrée pour voir une animation de chargement...")
    display.clear_screen()
    display.print_section("Animation de Chargement", 1)
    
    display.show_loading("Traitement en cours...")
    time.sleep(3)  # Simulation d'un traitement
    display.stop_loading()

def demo_charts(display: ConsoleDisplay):
    """Démonstration des graphiques."""
    display.clear_screen()
    display.print_header("Graphiques et Visualisation")
    
    # Générer des données de démo
    data = generate_sample_data(30)
    
    # Afficher un graphique en chandeliers
    display.print_section("Graphique en Chandeliers (OHLC)", 1)
    print("Chargement du graphique...")
    
    # Convertir les données pour le rendu
    candles = []
    for d in data:
        candles.append({
            'timestamp': d['timestamp'],
            'open': d['open'],
            'high': d['high'],
            'low': d['low'],
            'close': d['close'],
            'volume': d['volume']
        })
    
    # Afficher le graphique
    chart = display.render_candlestick(candles, width=80, height=15)
    print(chart)
    
    # Afficher un indicateur technique
    input("\nAppuyez sur Entrée pour voir un indicateur technique...")
    display.clear_screen()
    display.print_section("Indicateur Technique (RSI simulé)", 1)
    
    # Générer des valeurs RSI simulées (entre 0 et 100)
    rsi_values = [30 + random.random() * 40 for _ in range(80)]  # Entre 30 et 70
    
    # Afficher l'indicateur
    indicator = display.render_indicator(
        rsi_values,
        width=80,
        height=10,
        upper_bound=70,
        lower_bound=30,
        overbought=70,
        oversold=30
    )
    print("RSI (Relative Strength Index):")
    print(indicator)

def demo_dashboard(display: ConsoleDisplay):
    """Démonstration du tableau de bord interactif."""
    display.clear_screen()
    display.print_header("Tableau de Bord Interactif")
    
    print("Préparation du tableau de bord...")
    
    # Données de démo
    data = generate_sample_data(30)
    prices = [d['close'] for d in data]
    
    # Créer le tableau de bord
    dashboard = display.create_dashboard("Tableau de Bord du Trading", refresh_rate=1.0)
    
    # Widget 1: Prix actuel
    def get_price_widget() -> str:
        last_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else last_price
        change = ((last_price - prev_price) / prev_price) * 100
        change_icon = "🔼" if change >= 0 else "🔽"
        change_color = "GREEN" if change >= 0 else "RED"
        
        return (
            f"Prix actuel: {last_price:.2f} USD\n"
            f"Variation: {display.COLORS[change_color]}{change_icon} {abs(change):.2f}%{display.COLORS['ENDC']}"
        )
    
    # Widget 2: Graphique des prix (simplifié)
    def get_chart_widget() -> str:
        # Utiliser uniquement les 20 dernières valeurs pour le widget
        recent_data = data[-20:]
        if not recent_data:
            return "Aucune donnée disponible"
        
        # Créer une représentation simplifiée
        min_price = min(d['low'] for d in recent_data)
        max_price = max(d['high'] for d in recent_data)
        price_range = max(0.01, max_price - min_price)
        
        # Créer un graphique ASCII simple
        height = 10
        width = min(40, len(recent_data) * 2)
        chart = [[' ' for _ in range(width)] for _ in range(height)]
        
        for i, d in enumerate(recent_data[-width:]):
            # Calculer les positions pour OHLC
            open_pos = int(((d['open'] - min_price) / price_range) * (height - 1))
            close_pos = int(((d['close'] - min_price) / price_range) * (height - 1))
            high_pos = int(((d['high'] - min_price) / price_range) * (height - 1))
            low_pos = int(((d['low'] - min_price) / price_range) * (height - 1))
            
            # Ajuster les positions pour l'affichage (origine en bas à gauche)
            open_pos = height - 1 - open_pos
            close_pos = height - 1 - close_pos
            high_pos = height - 1 - high_pos
            low_pos = height - 1 - low_pos
            
            # Tracer la ligne haute-basse
            for y in range(min(high_pos, low_pos), max(high_pos, low_pos) + 1):
                if 0 <= y < height:
                    chart[y][i] = '│'
            
            # Tracer le corps de la bougie
            if open_pos == close_pos:
                # Ligne horizontale pour les bougies neutres
                if 0 <= open_pos < height:
                    chart[open_pos][i] = '─'
            else:
                # Bougie pleine pour hausse/baisse
                for y in range(min(open_pos, close_pos), max(open_pos, close_pos) + 1):
                    if 0 <= y < height:
                        chart[y][i] = '█' if open_pos < close_pos else '░'  # Plein pour hausse, hachuré pour baisse
        
        # Convertir en chaîne
        chart_lines = []
        for row in chart:
            chart_lines.append(''.join(row))
        
        return '\n'.join(chart_lines)
    
    # Widget 3: Derniers signaux
    signals = [
        {"paire": "BTC/USD", "signal": "ACHAT", "prix": 42500.50, "confiance": 0.87},
        {"paire": "ETH/USD", "signal": "VENTE", "prix": 2350.75, "confiance": 0.65},
        {"paire": "XRP/USD", "signal": "NEUTRE", "prix": 0.5678, "confiance": 0.42}
    ]
    
    def get_signals_widget() -> str:
        lines = ["Derniers signaux:"]
        for sig in signals:
            signal_icon = "🟢" if sig["signal"] == "ACHAT" else "🔴" if sig["signal"] == "VENTE" else "⚪"
            lines.append(f"{signal_icon} {sig['paire']}: {sig['signal']} à {sig['prix']:.4f} (confiance: {sig['confiance']*100:.0f}%)")
        return '\n'.join(lines)
    
    # Ajouter les widgets au tableau de bord
    dashboard.add_widget("Prix Actuel", get_price_widget, interval=1.0)
    dashboard.add_widget("Graphique des Prix", get_chart_widget, interval=2.0)
    dashboard.add_widget("Signaux", get_signals_widget, interval=5.0)
    
    # Démarrer le tableau de bord
    print("Le tableau de bord va démarrer. Utilisez les commandes suivantes:")
    print("- Q: Quitter")
    print("- R: Rafraîchir")
    print("- H: Aide")
    input("\nAppuyez sur Entrée pour continuer...")
    
    display.start_dashboard()

def main():
    """Fonction principale de la démonstration."""
    # Initialiser l'affichage avec le thème par défaut
    display = ConsoleDisplay(theme=Theme.DEFAULT)
    
    try:
        # Démarrer la démonstration
        demo_basic_features(display)
        
        input("\nAppuyez sur Entrée pour passer aux graphiques...")
        demo_charts(display)
        
        input("\nAppuyez sur Entrée pour accéder au tableau de bord interactif...")
        demo_dashboard(display)
        
    except KeyboardInterrupt:
        print("\nDémonstration interrompue par l'utilisateur.")
    except Exception as e:
        display.print_status(f"Une erreur est survenue: {str(e)}", "error")
    finally:
        # Nettoyage
        display.logger.stop()
        print("\nDémonstration terminée.")

if __name__ == "__main__":
    main()
