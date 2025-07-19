# 👋 Vue d'ensemble

Bienvenue dans le guide utilisateur de Kraken_Bot. Ce guide vous aidera à comprendre comment utiliser efficacement la plateforme pour vos opérations de trading automatisé.

## Fonctionnalités principales

- **Trading automatisé** : Exécution automatique des stratégies de trading
- **Stratégies intégrées** : Plusieurs stratégies prédéfinies disponibles
- **Gestion des risques** : Paramètres avancés de gestion du capital et des risques
- **Backtesting** : Testez vos stratégies sur des données historiques
- **Monitoring en temps réel** : Suivez les performances de vos trades
- **Rapports détaillés** : Analysez les performances de votre portefeuille

## Interface utilisateur

Kraken_Bot propose plusieurs interfaces :

1. **Interface en ligne de commande (CLI)** : Pour les utilisateurs avancés
2. **Interface web** : Tableau de bord intuitif (en développement)
3. **API REST** : Pour l'intégration avec d'autres systèmes

## Premier démarrage

1. **Lancez le bot** :
   ```bash
   python main.py
   ```

2. **Interface interactive** :
   ```
   ====================================
   KRAKEN BOT - TRADING TERMINAL
   ====================================
   
   [1] Démarrer le trading
   [2] Configurer les stratégies
   [3] Afficher le portefeuille
   [4] Voir les performances
   [5] Paramètres
   [6] Quitter
   
   Votre choix :
   ```

3. **Commandes principales** :
   - `start` : Démarrer le trading
   - `stop` : Arrêter le trading
   - `status` : Afficher l'état actuel
   - `help` : Afficher l'aide

## Configuration rapide

1. **Activer une stratégie** :
   ```bash
   python manage_config.py --strategy momentum --enable
   ```

2. **Configurer les paramètres** :
   ```bash
   python manage_config.py --set trading.max_open_trades=5
   ```

3. **Vérifier la configuration** :
   ```bash
   python manage_config.py --show
   ```

## Surveillance des performances

Utilisez les commandes suivantes pour surveiller les performances :

```bash
# Voir les positions ouvertes
python manage_config.py --positions

# Voir l'historique des trades
python manage_config.py --history

# Générer un rapport de performance
python manage_config.py --report
```

## Bonnes pratiques

1. **Commencez en mode simulation** : Activez le mode `DRY_RUN` avant de trader avec de l'argent réel
2. **Testez vos stratégies** : Utilisez le backtesting avant de passer en production
3. **Surveillez régulièrement** : Même en mode automatisé, gardez un œil sur les performances
4. **Gérez vos risques** : Ne risquez jamais plus que ce que vous pouvez vous permettre de perdre

## Dépannage

Si vous rencontrez des problèmes :

1. **Vérifiez les logs** : `tail -f logs/trading_bot.log`
2. **Vérifiez la connexion** : `python test_connection.py`
3. **Consultez les FAQ** : [FAQ](../faq.md)

## Prochaines étapes

- [Stratégies disponibles](strategies.md)
- [Gestion des risques](risk_management.md)
- [Configuration avancée](../getting_started/configuration.md)
