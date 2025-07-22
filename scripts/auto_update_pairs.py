import requests
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auto_update_pairs')

# Chemins de configuration
CONFIG_PATH = Path('config/config.yaml')
API_URL = 'https://api.kraken.com/0/public/AssetPairs'

# Importer la fonction de normalisation depuis le module core
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.core.services.pair_selector.types import normalize_pair_input

def load_config() -> Dict[str, Any]:
    """Charge le fichier de configuration YAML.
    
    Returns:
        Dictionnaire contenant la configuration
    """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return {}

def save_config(config: Dict[str, Any]) -> bool:
    """Sauvegarde la configuration dans le fichier YAML.
    
    Args:
        config: Dictionnaire de configuration à sauvegarder
        
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        # Créer le répertoire s'il n'existe pas
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, sort_keys=False, allow_unicode=True)
        logger.info(f"Configuration sauvegardée dans {CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
        return False

def get_kraken_pairs() -> List[str]:
    """Récupère et normalise les paires de trading disponibles sur Kraken.
    
    Returns:
        Liste des paires normalisées, triées par liquidité décroissante
    """
    try:
        logger.info("Récupération des paires Kraken...")
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if 'result' not in data:
            logger.error("Réponse inattendue de l'API Kraken: %s", data)
            return []
            
        # Récupérer les paires avec leurs métadonnées
        pairs_data = data['result']
        pairs_with_volume = []
        
        for pair_name, pair_info in pairs_data.items():
            try:
                # Ignorer les paires avec un .d (marché dérivé)
                if '.d' in pair_name:
                    continue
                    
                # Extraire le volume sur 24h si disponible
                volume = float(pair_info.get('volume_24h', 0))
                
                # Normaliser le nom de la paire
                normalized = normalize_pair_input(pair_name)
                
                pairs_with_volume.append({
                    'name': normalized,
                    'volume': volume,
                    'original_name': pair_name
                })
                
            except (ValueError, AttributeError) as e:
                logger.warning("Paire ignorée %s: %s", pair_name, e)
        
        # Trier par volume décroissant
        sorted_pairs = sorted(
            pairs_with_volume, 
            key=lambda x: x['volume'], 
            reverse=True
        )
        
        # Extraire uniquement les noms normalisés
        normalized_pairs = [p['name'] for p in sorted_pairs]
        
        logger.info(
            "%d paires récupérées et normalisées (top 5 par volume: %s)", 
            len(normalized_pairs),
            ', '.join(normalized_pairs[:5]) if normalized_pairs else 'aucune'
        )
        
        return normalized_pairs
        
    except requests.RequestException as e:
        logger.error("Erreur de requête vers l'API Kraken: %s", e)
        return []
    except Exception as e:
        logger.exception("Erreur inattendue lors de la récupération des paires")
        return []

def update_pairs(max_pairs: int = 50) -> None:
    """
    Met à jour la configuration avec les paires disponibles sur Kraken.
    
    Args:
        max_pairs: Nombre maximum de paires à conserver (les plus liquides)
    """
    try:
        # Récupérer les paires normalisées triées par liquidité
        all_pairs = get_kraken_pairs()
        if not all_pairs:
            logger.error("Aucune paire valide n'a pu être récupérée")
            return
            
        # Limiter le nombre de paires si nécessaire
        pairs = all_pairs[:max_pairs] if max_pairs > 0 else all_pairs
        
        # Charger la configuration existante
        config = load_config()
        
        # Vérifier si les paires ont changé
        old_pairs = set(config.get('trading', {}).get('pairs', []))
        new_pairs_set = set(pairs)
        
        if old_pairs == new_pairs_set:
            logger.info("Aucun changement détecté dans les paires disponibles")
            return
            
        # Initialiser la section trading si elle n'existe pas
        if 'trading' not in config:
            config['trading'] = {}
        
        # Mettre à jour les paires dans la configuration
        previous_count = len(config['trading'].get('pairs', []))
        config['trading']['pairs'] = pairs
        
        # Ajouter des métadonnées utiles
        config['trading']['last_updated'] = datetime.now().isoformat()
        config['trading']['total_pairs'] = len(pairs)
        
        # Sauvegarder la configuration
        if save_config(config):
            added = len(new_pairs_set - old_pairs)
            removed = len(old_pairs - new_pairs_set)
            
            logger.info(
                "✅ Configuration mise à jour avec %d paires (%d ajoutées, %d supprimées)",
                len(pairs), added, removed
            )
            
            if added > 0:
                logger.info("Nouvelles paires: %s", 
                          ", ".join(sorted(new_pairs_set - old_pairs)[:10]) + 
                          ("..." if added > 10 else ""))
        else:
            logger.error("❌ Échec de la mise à jour de la configuration")
            
    except Exception as e:
        logger.error("Erreur lors de la mise à jour des paires: %s", str(e), exc_info=True)

if __name__ == "__main__":
    update_pairs()