# Stratégies de Trading

Ce répertoire contient la documentation détaillée des différentes stratégies de trading implémentées dans le bot.

## Sommaire

1. [Stratégie de Breakout](BREAKOUT_STRATEGY.md)
   - Détection des niveaux de support/résistance
   - Entrées sur cassures de range
   - Gestion des faux signaux

2. [Stratégie de Swing Trading](SWING_STRATEGY.md)
   - Identification des tendances intermédiaires
   - Gestion des positions sur plusieurs jours
   - Optimisation du ratio risque/rendement

3. [Stratégies Multi-Marchés](MULTI_MARKET_STRATEGY.md)
   - Corrélations entre actifs
   - Allocation de capital entre marchés
   - Gestion des risques croisés

4. [Stratégies Multi-Timeframes](multi_market_timeframe.md)
   - Analyse hiérarchique des tendances
   - Synchronisation des signaux
   - Optimisation des entrées/sorties

## Comment Choisir une Stratégie

Le choix d'une stratégie dépend de plusieurs facteurs :

- **Horizon de trading** : court, moyen ou long terme
- **Disponibilité** : temps de surveillance requis
- **Tolérance au risque** : volatilité acceptée
- **Capital disponible** : marge de manœuvre nécessaire

## Développement de Nouvelles Stratégies

Pour développer une nouvelle stratégie :

1. Créez un nouveau fichier `NOM_STRATEGIE.md`
2. Implémentez la classe de stratégie dans `src/strategies/`
3. Ajoutez des tests unitaires
4. Documentez la stratégie selon le modèle existant

## Bonnes Pratiques

- Tester systématiquement sur données historiques
- Valider avec un compte démo avant le déploiement
- Surveiller régulièrement les performances
- Adapter les paramètres aux conditions de marché
