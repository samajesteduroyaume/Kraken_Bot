"""
Script d'automatisation de l'entraînement des modèles ML.
Ce script vérifie périodiquement si les modèles ont besoin d'être réentraînés
et les met à jour si nécessaire.
"""

import asyncio
import os
import time
from typing import Optional

# Configuration du logging
from kraken_bot.utils.logger import setup_logger
logger = setup_logger('auto_trainer')

# Import local
from kraken_bot.core.database import db_manager, init_db, close_db
from kraken_bot.ml.predictor import MLPredictor
from config.settings import ML_CONFIG, TRADING_PAIRS

class AutoTrainer:
    """Classe pour gérer l'entraînement automatique des modèles."""
    
    def __init__(self):
        """Initialise l'auto-trainer avec le prédicteur ML."""
        self.ml_predictor = MLPredictor(db_manager=db_manager)
        self.running = False
        self.check_interval = ML_CONFIG['check_interval']
        self.pairs = TRADING_PAIRS
        
    async def check_and_train(self, pair: str) -> Optional[float]:
        """Vérifie si le modèle a besoin d'être réentraîné et le fait si nécessaire.
        
        Args:
            pair: La paire de trading à vérifier
            
        Returns:
            Le score du modèle ou None si l'entraînement a échoué
        """
        try:
            # Vérifier et entraîner si nécessaire
            score = await self.ml_predictor.train(pair)
            
            if score is not None:
                logger.info(f"Modèle pour {pair} entraîné avec succès (score: {score:.4f})")
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du modèle pour {pair}: {e}", exc_info=True)
            return None
    
    async def run(self):
        """Lance la boucle principale de vérification et d'entraînement."""
        self.running = True
        logger.info("Démarrage de l'auto-trainer...")
        
        try:
            # Initialiser la base de données
            await init_db()
            
            while self.running:
                start_time = time.time()
                
                # Vérifier et entraîner chaque paire
                for pair in self.pairs:
                    try:
                        await self.check_and_train(pair)
                        # Petite pause entre chaque paire pour éviter de surcharger l'API
                        await asyncio.sleep(5)
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de {pair}: {e}", exc_info=True)
                
                # Calculer le temps d'attente avant la prochaine vérification
                elapsed = time.time() - start_time
                wait_time = max(0, self.check_interval - elapsed)
                
                logger.info(f"Prochaine vérification dans {wait_time/60:.1f} minutes...")
                await asyncio.sleep(wait_time)
                
        except asyncio.CancelledError:
            logger.info("Arrêt demandé, fermeture propre...")
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}", exc_info=True)
        finally:
            self.running = False
            await close_db()
            logger.info("Auto-trainer arrêté")

async def main():
    """Fonction principale."""
    # Créer les dossiers nécessaires
    os.makedirs(ML_CONFIG['model_dir'], exist_ok=True)
    
    # Démarrer l'auto-trainer
    trainer = AutoTrainer()
    
    try:
        await trainer.run()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    finally:
        trainer.running = False
        await close_db()

if __name__ == "__main__":
    asyncio.run(main())
