"""
Module pour la sélection des paires de trading basée sur des critères techniques.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import heapq

from src.core.api.kraken import KrakenAPI
from src.core.simulation_mode import SimulationConfig

logger = logging.getLogger(__name__)


class PairSelector:
    """Classe pour sélectionner les paires de trading basée sur des critères techniques."""

    def __init__(
        self,
        api: KrakenAPI,
        config: SimulationConfig,
        max_pairs: int = 50,
        min_volume: float = 100000,
        volatility_window: int = 20,
        momentum_window: int = 14,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """Initialise le sélecteur de paires."""
        self.api = api
        self.config = config
        self.max_pairs = max_pairs
        self.min_volume = min_volume
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.executor = executor or ThreadPoolExecutor(max_workers=10)
        # self.available_pairs = config.available_pairs  # SUPPRIMÉ
        self.analysis_results: Dict[str, Any] = {}
        self.pair_metrics: Dict[str, Dict] = {}
        self.last_update: datetime = datetime.now(timezone.utc)
        self.update_frequency: timedelta = timedelta(hours=1)

    async def _load_pairs_from_api(self) -> set:
        """
        Charge les paires disponibles depuis l'API Kraken.
        
        Returns:
            Ensemble des noms de paires disponibles
        """
        try:
            logger.info("Récupération des paires disponibles sur Kraken...")
            available_pairs = await self.api.get_asset_pairs()
            
            if not available_pairs or not isinstance(available_pairs, dict):
                logger.error("Aucune paire disponible ou format de réponse invalide de l'API Kraken")
                return set()
            # Correction : utiliser le champ 'result' si présent
            if 'result' in available_pairs:
                available_pairs = available_pairs['result']
            # Extraire uniquement les noms des paires
            pair_names = set(available_pairs.keys())
            logger.info(f"{len(pair_names)} paires récupérées de l'API Kraken")
            return pair_names
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des paires depuis l'API: {str(e)}", exc_info=True)
            return set()

    async def _analyze_pair(self, pair: str) -> Dict[str, Any]:
        """Analyse une paire individuelle."""
        try:
            async with self.api:
                # DEBUG : Afficher le mapping altname pour la paire
                await self.api._ensure_pair_altname_map()
                alt_name = self.api.pair_altname_map.get(pair, 'Non trouvé')
                logger.debug(f"[ANALYSE] Début analyse pour {pair} (altname: {alt_name})")
                
                # Récupérer les données OHLC
                try:
                    ohlc_data = await self.api.get_ohlc_data(
                        pair=pair,
                        interval=1440,  # 1 jour
                        since=int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
                    )
                    logger.debug(f"[ANALYSE] Données OHLC récupérées pour {pair}: {len(ohlc_data.get(pair, []))} entrées")
                except Exception as e:
                    logger.warning(f"[ANALYSE] Erreur récupération OHLC pour {pair}: {str(e)}")
                    # Essayer avec l'altname si disponible
                    if alt_name and alt_name != 'Non trouvé' and alt_name != pair:
                        logger.debug(f"[ANALYSE] Tentative avec l'altname: {alt_name}")
                        try:
                            ohlc_data = await self.api.get_ohlc_data(
                                pair=alt_name,
                                interval=1440,
                                since=int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
                            )
                            logger.debug(f"[ANALYSE] Données OHLC récupérées avec altname pour {pair}: {len(ohlc_data.get(alt_name, []))} entrées")
                        except Exception as e2:
                            logger.error(f"[ANALYSE] Échec avec l'altname {alt_name}: {str(e2)}")
                            return {}
                    else:
                        return {}

                # Vérifier si nous avons des données pour cette paire
                data_key = pair if pair in ohlc_data else alt_name if alt_name in ohlc_data else None
                if not data_key or not ohlc_data.get(data_key):
                    logger.warning(f"[ANALYSE] Aucune donnée OHLC pour {pair} (clé: {data_key})")
                    return {}

                # Convertir les valeurs OHLC en nombres
                try:
                    ohlc_data[data_key] = [
                        [float(val) if i > 0 else val for i, val in enumerate(row)]
                        for row in ohlc_data[data_key]
                        if len(row) >= 8  # Vérifier que la ligne a suffisamment d'éléments
                    ]
                    
                    if not ohlc_data[data_key]:
                        logger.warning(f"[ANALYSE] Aucune donnée valide après conversion pour {pair}")
                        return {}

                    # Calculer les indicateurs techniques
                    df = pd.DataFrame(
                        ohlc_data[data_key],
                        columns=[
                            'time',
                            'open',
                            'high',
                            'low',
                            'close',
                            'vwap',
                            'volume',
                            'count'
                        ]
                    )
                    
                    # Vérifier que nous avons des données valides
                    if df.empty:
                        logger.warning(f"[ANALYSE] DataFrame vide pour {pair}")
                        return {}
                        
                    # Convertir les colonnes numériques
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Supprimer les lignes avec des valeurs manquantes
                    df = df.dropna(subset=numeric_cols)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    if df.empty:
                        logger.warning(f"[ANALYSE] Plus de données après nettoyage pour {pair}")
                        return {}
                        
                except Exception as e:
                    logger.error(f"[ANALYSE] Erreur lors du traitement des données OHLC pour {pair}: {str(e)}", exc_info=True)
                    return {}
                df.set_index('time', inplace=True)
                if df.empty:
                    logger.error(f"DataFrame OHLC vide pour la paire {pair}")
                    print(f"[ERROR] DataFrame OHLC vide pour la paire {pair}")
                    return {}
                # Calculer le momentum
                momentum = df['close'].pct_change(
                    self.momentum_window).iloc[-1]
                # Calculer la volatilité
                returns = df['close'].pct_change()
                volatility = returns.rolling(
                    window=self.volatility_window).std().iloc[-1]
                # Calculer le volume moyen
                avg_volume = df['volume'].rolling(
                    window=self.volatility_window).mean().iloc[-1]
                # Calculer le score basé sur momentum, volatilité et volume
                score = (momentum * 0.4) - \
                    (volatility * 0.3) + (avg_volume * 0.3)
                # Créer le résultat
                pair_result = {
                    'pair': pair,
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume': avg_volume,
                    'score': score
                }
                # Mettre à jour les métriques
                self.pair_metrics[pair] = {
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume': avg_volume,
                    'score': score
                }
            return {
                'pair': pair,
                'score': float(score),
                'volatility': float(volatility),
                'momentum': float(momentum),
                'volume': float(df['volume'].mean()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(
                f"Erreur lors de l'analyse de la paire {pair}: {str(e)}",
                exc_info=True)
            return {}

    def _calculate_score(
            self,
            volatility: float,
            momentum: float,
            volume: float) -> float:
        """Calcule le score de la paire."""
        try:
            # Normaliser les valeurs avec des bornes plus réalistes
            # Volatilité : 0-1 (1 pour très stable, 0 pour très volatile)
            # Seuil de volatilité maximum : 1.0 (100%)
            norm_volatility = max(0, min(1, 1 - (min(1.0, volatility))))

            # Momentum : -1 à 1 -> 0 à 1
            # Seuil de momentum maximum : 1.0 (100%)
            norm_momentum = max(0, min(1, (momentum + 1) / 2))

            # Volume : normalisé entre 0 et 1
            # Utiliser un volume de référence plus réaliste (100000 USD)
            norm_volume = max(0, min(1, volume / 100000))

            # Pondération des facteurs avec des coefficients plus équilibrés
            score = (
                0.3 * norm_volatility +  # 30% pour la stabilité
                0.4 * norm_momentum +    # 40% pour la tendance
                0.3 * norm_volume        # 30% pour le volume
            )

            # Ajuster le score pour éviter les scores trop faibles
            score = max(0.1, score)  # Score minimum de 0.1

            return min(max(score, 0), 1)

        except Exception as e:
            logger.error(f"Erreur lors du calcul du score: {str(e)}")
            return 0.1  # Score minimum par défaut en cas d'erreur

    def _normalize_pair_name(self, base: str, quote: str) -> List[str]:
        """Génère les formats de noms de paires normalisés pour une paire donnée.
        
        Args:
            base: La devise de base (ex: 'XBT', 'ETH')
            quote: La devise de cotation (ex: 'EUR', 'USD')
            
        Returns:
            Liste des formats normalisés possibles pour la paire
        """
        formats = []
        
        # Format standard avec X et Z (XBTZEUR)
        formats.append(f"X{base}Z{quote}")
        
        # Format avec X sans Z (XBTUSD)
        formats.append(f"X{base}{quote}")
        
        # Format spécial pour BTC (XXBT)
        if base == 'BTC':
            formats.append(f"XXBTZ{quote}")
            formats.append(f"XXBT{quote}")
        
        # Format spécial pour ETH (XETH)
        if base == 'ETH':
            formats.append(f"XETHZ{quote}")
            formats.append(f"XETH{quote}")
        
        # Format inversé pour certaines paires (ex: USDT-BTC)
        formats.append(f"X{quote}Z{base}")
        formats.append(f"X{quote}{base}")
        
        return list(set(formats))  # Éliminer les doublons

    def _get_configured_pairs(self) -> set:
        """
        Récupère l'ensemble des paires configurées dans le fichier de configuration.
        
        Returns:
            Ensemble des paires normalisées pour l'API Kraken
        """
        configured_pairs = set()
        
        if not hasattr(self.config, 'available_pairs') or not self.config.available_pairs:
            logger.warning("Aucune paire configurée dans la configuration")
            return configured_pairs
            
        logger.info(f"Paires configurées dans config.yaml: {self.config.available_pairs}")
        
        for pair in self.config.available_pairs:
            try:
                if '/' in pair:
                    base, quote = pair.split('/')
                    normalized = self._normalize_pair_name(base, quote)
                    logger.debug(f"Paire normalisée: {pair} -> {normalized}")
                    configured_pairs.update(normalized)
                else:
                    logger.debug(f"Paire au format Kraken: {pair}")
                    configured_pairs.add(pair)
            except Exception as e:
                logger.error(f"Erreur lors de la normalisation de la paire {pair}: {str(e)}", exc_info=True)
        
        logger.info(f"Total des paires configurées normalisées: {len(configured_pairs)}")
        return configured_pairs

    async def get_valid_pairs(
            self,
            min_score: float = 0.0,  # Valeur par défaut abaissée pour le debug
            max_pairs: Optional[int] = None) -> List[str]:
        """
        Récupère et filtre les paires de trading valides depuis l'API Kraken.
        
        Args:
            min_score: Score minimum pour qu'une paire soit considérée comme valide
            max_pairs: Nombre maximum de paires à retourner (optionnel)
            
        Returns:
            Liste des paires valides triées par score décroissant
        """
        try:
            logger.info("\n=== DÉBUT DE LA SÉLECTION DES PAIRES ===")
            logger.info(f"Score minimum requis: {min_score}")
            
            # Récupérer les paires configurées
            logger.info("\n1. Récupération des paires configurées...")
            configured_pairs = self._get_configured_pairs()
            if not configured_pairs:
                logger.error("❌ Aucune paire valide trouvée dans la configuration")
                return []
            logger.info(f"✅ {len(configured_pairs)} paires configurées: {configured_pairs}")
                
            # Charger les paires disponibles depuis l'API
            logger.info("\n2. Chargement des paires disponibles depuis l'API Kraken...")
            available_pairs = await self._load_pairs_from_api()
            if not available_pairs:
                logger.error("❌ Impossible de charger les paires depuis l'API Kraken")
                return []
            logger.info(f"✅ {len(available_pairs)} paires disponibles sur Kraken")
            
            # Afficher un échantillon des paires disponibles pour le débogage
            sample_size = min(5, len(available_pairs))
            sample_pairs = list(available_pairs)[:sample_size]
            logger.debug(f"   Exemple de paires disponibles: {sample_pairs}...")
            
            print('DEBUG: PAIR LOGS')
            logger.info(f"\n=== LISTE COMPLÈTE DES PAIRES DISPONIBLES SUR KRAKEN ===")
            logger.info(
                f"{sorted(list(available_pairs))}"
            )
            logger.info(f"\n=== LISTE DES PAIRES CONFIGURÉES (normalisées) ===")
            logger.info(
                f"{sorted(list(configured_pairs))}"
            )
            if not available_pairs:
                print('DEBUG: AUCUNE PAIRE KRAKEN DISPONIBLE !')

            logger.info("\n3. Filtrage des paires configurées disponibles sur Kraken...")
            
            # Filtrer et analyser les paires configurées qui sont disponibles sur Kraken
            valid_pairs = []
            for pair in configured_pairs:
                # Vérifier la correspondance exacte d'abord
                if pair in available_pairs:
                    logger.info(f"✅ Correspondance exacte trouvée pour: {pair}")
                    matched_pair = pair
                # Correspondance insensible à la casse
                elif pair.upper() in {p.upper() for p in available_pairs}:
                    matched_pair = next(p for p in available_pairs if p.upper() == pair.upper())
                    logger.info(f"ℹ️  Correspondance insensible à la casse: {pair} -> {matched_pair}")
                # Mapping intelligent : ignorer les préfixes/suffixes Kraken
                else:
                    # Ex : XETHZUSD -> ETH/USD
                    def strip_kraken(s):
                        return s.replace('X', '').replace('Z', '').replace('/', '').upper()
                    base, quote = '', ''
                    if '/' in pair:
                        base, quote = pair.split('/')
                    else:
                        # Essai de split automatique
                        for q in ['USD', 'EUR', 'USDT', 'USDC', 'BTC', 'XBT', 'ETH']:
                            if pair.endswith(q):
                                base, quote = pair[:-len(q)], q
                                break
                    found = None
                    for ap in available_pairs:
                        ap_stripped = strip_kraken(ap)
                        if base and quote and (base.upper() + quote.upper() == ap_stripped):
                            found = ap
                            break
                    if found:
                        matched_pair = found
                        logger.info(f"🤖 Correspondance intelligente trouvée pour: {pair} -> {matched_pair}")
                    else:
                        continue
                # Analyser la paire correspondante
                try:
                    logger.info(f"🔍 Analyse de la paire: {matched_pair}")
                    pair_metrics = await self._analyze_pair(matched_pair)
                    if not pair_metrics:
                        continue
                    score = pair_metrics.get('score', 0)
                    if score >= min_score:
                        valid_pairs.append((matched_pair, pair_metrics))
                        logger.info(f"✅ Paire valide ajoutée: {matched_pair} (score: {score:.4f})")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'analyse de la paire {matched_pair}: {str(e)}", exc_info=True)
                    continue
            
            # Trier par score décroissant
            valid_pairs.sort(key=lambda x: x[1].get('score', 0), reverse=True)
            
            # Limiter le nombre de paires si nécessaire
            if max_pairs is not None and len(valid_pairs) > max_pairs:
                valid_pairs = valid_pairs[:max_pairs]
            
            # Mettre à jour le cache
            self.analysis_results = {pair: metrics for pair, metrics in valid_pairs}
            self.last_update = datetime.now(timezone.utc)
            
            # Journalisation des résultats
            logger.info("\n=== RÉSULTATS DE LA SÉLECTION ===")
            if not valid_pairs:
                logger.warning("❌ Aucune paire valide trouvée avec les critères actuels")
                logger.info("Vérifiez que :")
                logger.info("- Les paires configurées existent bien sur Kraken")
                logger.info("- Les scores minimaux ne sont pas trop élevés")
                logger.info("- La connexion à l'API Kraken fonctionne correctement")
                return []
                
            logger.info(f"✅ {len(valid_pairs)} paires valides sélectionnées sur {len(configured_pairs)} configurées")
            logger.info("\n=== PAIRES SÉLECTIONNÉES ===")
            for i, (pair, metrics) in enumerate(valid_pairs, 1):
                score = metrics.get('score', 0)
                volatility = metrics.get('volatility', 0) * 100  # En pourcentage
                momentum = metrics.get('momentum', 0) * 100      # En pourcentage
                volume = metrics.get('volume', 0)
                
                logger.info(
                    f"{i:2d}. {pair:10} | "
                    f"Score: {score:6.2f} | "
                    f"Vol: {volatility:5.2f}% | "
                    f"Mom: {momentum:6.2f}% | "
                    f"Vol: {volume:12.2f} {pair[-3:] if len(pair) >= 3 else '  '}"
                )

            logger.info(f"\n✅ {len(valid_pairs)} paires sélectionnées sur {len(configured_pairs)} configurées")
            return [pair for pair, _ in valid_pairs]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des paires valides: {str(e)}", exc_info=True)
            return []

    def get_pair_metrics(self, pair: str) -> Dict[str, Any]:
        """Retourne les métriques d'une paire spécifique."""
        return self.pair_metrics.get(pair, {})

    def get_top_pairs(self, n: int = 10) -> List[Tuple[str, float]]:
        """Retourne les n meilleures paires avec leurs scores."""
        return heapq.nlargest(
            n,
            self.analysis_results.items(),
            key=lambda x: x[1].get('score', 0)
        )
