"""
Package services - Contient les services du bot de trading Kraken.
"""

from .auto_train import AutoTrainer, main as auto_train_main

__all__ = ['AutoTrainer', 'auto_train_main']
