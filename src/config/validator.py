from typing import List
import os
import re
from .types import TradingConfig, RiskProfile
from .risk_profiles import RISK_PROFILES


class ConfigValidator:
    """
    Valide les configurations du bot.

    Cette classe fournit des méthodes pour valider les configurations principales du bot,
    notamment les variables d'environnement, la configuration de trading et les profils de risque.
    """

    # Variables d'environnement requises
    REQUIRED_ENV_VARS = {
        "database": [
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB"
        ],
        "api": [
            "KRAKEN_API_KEY",
            "KRAKEN_API_SECRET"
        ],
        "trading": [
            "TRADING_ENABLED",
            "STRATEGY",
            "TIMEFRAME",
            "RISK_PROFILE"
        ],
        "ml": [
            "ML_ENABLED",
            "ML_MODEL_TYPE",
            "ML_LOOKBACK_DAYS",
            "ML_TRAIN_VALIDATION_SPLIT",
            "ML_EPOCHS",
            "ML_BATCH_SIZE",
            "ML_LEARNING_RATE",
            "ML_DROPOUT_RATE"
        ]
    }

    # Expressions régulières pour la validation des formats
    TIMEFRAME_PATTERN = re.compile(r"^(\d+m|\d+h|\d+d)$")
    PAIR_PATTERN = re.compile(r"^[A-Z]+/[A-Z]+$")

    @staticmethod
    def validate_env_vars():
        """
        Vérifie que toutes les variables d'environnement requises sont présentes et valides.

        Raises:
            EnvironmentError: Si des variables d'environnement manquent ou sont invalides
        """
        errors = []

        # Vérifier les variables requises par catégorie
        for category, vars in ConfigValidator.REQUIRED_ENV_VARS.items():
            missing_vars = [var for var in vars if not os.getenv(var)]
            if missing_vars:
                errors.append(
                    f"Dans la catégorie '{category}', les variables suivantes sont manquantes: {', '.join(missing_vars)}")

        # Vérifier les formats spécifiques
        timeframe = os.getenv("TIMEFRAME")
        if timeframe:
            # Nettoyer les espaces et les commentaires
            timeframe = timeframe.split('#')[0].strip()
            if not ConfigValidator.TIMEFRAME_PATTERN.match(timeframe):
                errors.append(
                    f"Le format du timeframe '{timeframe}' est invalide. Format attendu: 1m, 5m, 15m, 1h, 4h, 1d")

        if errors:
            raise EnvironmentError("\n".join(errors))

    @staticmethod
    def validate_trading_config(config: TradingConfig) -> List[str]:
        """
        Valide la configuration de trading.

        Args:
            config: Configuration de trading à valider

        Returns:
            List[str]: Liste des erreurs de validation
        """
        errors = []

        # Vérification de base
        if not isinstance(config.get("enabled"), bool):
            errors.append("La configuration 'enabled' doit être un booléen")

        if not isinstance(config.get("pairs"), list):
            errors.append("La configuration 'pairs' doit être une liste")

        if not config.get("pairs"):
            errors.append("La liste des paires ne peut pas être vide")

        if not isinstance(config.get("initial_balance"), (int, float)):
            errors.append(
                "La configuration 'initial_balance' doit être un nombre")

        # Vérification des paires
        for pair in config.get("pairs", []):
            if not isinstance(pair, dict):
                errors.append(f"La paire {pair} doit être un dictionnaire")
                continue

            if not ConfigValidator.PAIR_PATTERN.match(pair.get("symbol", "")):
                errors.append(
                    f"Le format de la paire {pair.get('symbol')} est invalide. Format attendu: XBT/USD")

            if not isinstance(pair.get("min_size"), (int, float)
                              ) or pair.get("min_size") <= 0:
                errors.append(
                    f"La taille minimum de la paire {pair.get('symbol')} doit être un nombre positif")

        # Vérification du profil de risque
        risk_profile = config.get("profile")
        if not isinstance(risk_profile, dict):
            errors.append("Le profil de risque doit être un dictionnaire")
        else:
            profile_name = risk_profile.get("risk_level")
            if profile_name not in RISK_PROFILES:
                errors.append(
                    f"Le profil de risque '{profile_name}' est inconnu. Profils disponibles: {', '.join(RISK_PROFILES.keys())}")

        # Vérification des paramètres de stratégie
        if not isinstance(config.get("strategy"), str):
            errors.append("La stratégie doit être une chaîne de caractères")

        if not isinstance(config.get("max_trade_amount"), (int, float)) or config.get(
                "max_trade_amount") <= 0:
            errors.append(
                "Le montant maximum de trade doit être un nombre positif")

        return errors

    @staticmethod
    def validate_risk_profile(profile: RiskProfile) -> List[str]:
        """
        Valide un profil de risque.

        Args:
            profile: Profil de risque à valider

        Returns:
            List[str]: Liste des erreurs de validation
        """
        errors = []

        # Vérification des pourcentages
        for field in [
            "max_position_size",
            "stop_loss_percentage",
            "take_profit_percentage",
                "max_drawdown"]:
            value = profile.get(field)
            if not isinstance(value, (int, float)) or not (0 <= value <= 100):
                errors.append(f"{field} doit être un nombre entre 0 et 100")

        # Vérification des limites
        if profile.get("max_position_size") > 100:
            errors.append("max_position_size ne peut pas être supérieur à 100")

        if profile.get("stop_loss_percentage") >= profile.get(
                "take_profit_percentage"):
            errors.append(
                "take_profit_percentage doit être supérieur à stop_loss_percentage")

        return errors
