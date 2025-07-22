# 👋 Guide Utilisateur - Vue d'Ensemble

Bienvenue dans le guide utilisateur de Kraken_Bot. Cette documentation complète vous guidera à travers toutes les fonctionnalités de la plateforme pour optimiser votre expérience de trading automatisé.

## 🚀 Premiers Pas Rapides

### 1. Configuration Initiale
Avant de commencer, assurez-vous d'avoir :
- [ ] Installé Kraken_Bot (voir [Guide d'installation](../getting_started/installation.md))
- [ ] Configuré votre fichier `.env` avec vos clés API
- [ ] Vérifié les paramètres de base dans `config/config.yaml`

### 2. Lancement du Bot

#### Interface en Ligne de Commande (CLI)
```bash
# Mode standard
python main.py

# Mode démon (en arrière-plan)
python main.py --daemon

# Mode simulation uniquement
python main.py --dry-run
```

#### Interface Web (si activée)
```
http://localhost:8000
```

## 🎯 Fonctionnalités Principales

### 📊 Tableau de Bord
- Vue d'ensemble du portefeuille
- Performances en temps réel
- Graphiques interactifs
- Alertes et notifications

### 🤖 Stratégies de Trading
- **Stratégies intégrées** :
  - 📈 Momentum Trading
  - 🔄 Mean Reversion
  - 🎯 Breakout
  - 🧩 Grid Trading
  - 🔄 Swing Trading
- **Personnalisation** des paramètres
- Backtesting intégré
- Optimisation des paramètres

### 🛡️ Gestion des Risques
- Stop-loss dynamiques
- Take-profit progressifs
- Gestion de la taille de position
- Limites de perte quotidienne
- Protection contre la volatilité excessive

### 📊 Analyse et Rapports
- Analyse technique avancée
- Rapports de performance détaillés
- Journal des transactions
- Export des données au format CSV/Excel

## 🖥️ Interfaces Disponibles

### 1. Interface Web (Recommandée)
```
http://localhost:8000
```

**Fonctionnalités clés** :
- Tableau de bord personnalisable
- Gestion des stratégies en temps réel
- Visualisation des positions ouvertes
- Historique des transactions
- Alertes et notifications
- Configuration avancée

### 2. Interface en Ligne de Commande (CLI)
```bash
# Afficher l'aide
python main.py --help

# Lancer avec une configuration spécifique
python main.py --config config/prod_config.yaml

# Activer le mode debug
python main.py --debug

# Voir les logs en temps réel
tail -f logs/kraken_bot.log
```

### 3. API REST
Accédez à toutes les fonctionnalités via notre API REST complète :
```python
import requests
import hmac
import hashlib
import time

# Configuration de base
api_key = "votre_cle_api"
api_secret = "votre_secret_api"
base_url = "https://votre-serveur.com/api/v1"

# Exemple de requête d'authentifiée
def get_account_balance():
    nonce = str(int(time.time() * 1000))
    message = nonce + "/api/v1/account/balance"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha512
    ).hexdigest()
    
    headers = {
        'API-Key': api_key,
        'API-Signature': signature,
        'Content-Type': 'application/json'
    }
    
    response = requests.get(
        f"{base_url}/account/balance",
        headers=headers,
        json={"nonce": nonce}
    )
    return response.json()
```

## 🔍 Premiers Pas avec le Trading

### 1. Configuration d'une Stratégie
1. Accédez à l'onglet "Stratégies"
2. Cliquez sur "Ajouter une stratégie"
3. Sélectionnez le type de stratégie
4. Configurez les paramètres
5. Activez la stratégie

### 2. Surveillance des Performances
- Tableau de bord en temps réel
- Alertes personnalisables
- Rapports quotidiens par email
- Analyses hebdomadaires détaillées

### 3. Gestion des Risques
- Définissez vos limites de perte
- Activez les stops dynamiques
- Surveillez votre exposition au risque
- Ajustez en temps réel

## 📱 Application Mobile
Accédez à votre compte partout avec notre application mobile disponible sur :
- **App Store** (iOS)
- **Google Play** (Android)

**Fonctionnalités mobiles** :
- Notifications push instantanées
- Tableau de bord personnalisable
- Exécution rapide des ordres
- Alertes de marché personnalisées
- Accès aux rapports détaillés

## ❓ Support et Assistance

### 1. Centre d'Aide
Consultez notre [base de connaissances](../faq/README.md) pour des réponses aux questions courantes.

### 2. Support Technique
- **Email** : support@krakenbot.com
- **Chat en direct** : Disponible dans l'application
- **Téléphone** : +33 1 23 45 67 89 (Lun-Ven 9h-18h CET)

### 3. Communauté
Rejoignez notre communauté de traders :
- [Forum de discussion](https://community.krakenbot.com)
- [Groupe Telegram](https://t.me/krakenbot_fr)
- [YouTube](https://youtube.com/krakenbot) pour des tutoriels

## 🔄 Mises à Jour et Maintenance
- Mises à jour automatiques
- Fenêtres de maintenance annoncées à l'avance
- Historique des versions dans [CHANGELOG.md](../../CHANGELOG.md)

## 🔒 Sécurité
- Authentification à deux facteurs (2FA)
- Chiffrement de bout en bout
- Aucun retrait automatique possible
- Journalisation complète des activités

## 📚 Pour Aller Plus Loin
- [Guide avancé des stratégies](strategies.md)
- [Gestion avancée des risques](risk_management.md)
- [Documentation de l'API](../api_reference/rest_api.md)
- [Guide du développeur](../developer_guide/creating_strategies.md)

---

💡 **Conseil** : Consultez régulièrement nos mises à jour pour bénéficier des dernières fonctionnalités et améliorations de sécurité.
- Tableau de bord interactif
- Configuration visuelle des stratégies
- Visualisation des graphiques en temps réel
- Gestion des positions

### 2. Interface Ligne de Commande (CLI)

**Commandes principales** :
```bash
# Démarrer le trading
python main.py start

# Arrêter le trading
python main.py stop

# Statut actuel
python main.py status

# Voir le portefeuille
python main.py portfolio

# Lancer un backtest
python main.py backtest --strategy=momentum --days=30
```

### 3. API REST
Documentation complète disponible dans la [Référence API](../api_reference/overview.md).

## 🏁 Démarrer son Premier Trade

1. **Vérifiez la configuration** :
   ```bash
   python main.py check-config
   ```

2. **Lancez en mode simulation** :
   ```bash
   python main.py --dry-run
   ```

3. **Analysez les performances** dans l'interface web ou avec :
   ```bash
   python main.py performance
   ```

4. **Passez en mode réel** une fois satisfait :
   ```bash
   python main.py start
   ```

## 📱 Application Mobile (Bêta)

Accédez à votre compte depuis n'importe où avec notre application mobile :
- **iOS** : Disponible sur l'App Store
- **Android** : Disponible sur Google Play

## 📞 Support

Besoin d'aide ? Consultez :
- [FAQ](../faq.md)
- [Dépannage](../troubleshooting/common_issues.md)
- [Discord communautaire](https://discord.gg/votreserveur)
- Support par email : support@krakenbot.com

## 📚 Prochaines Étapes

- [ ] Configurer votre première stratégie
- [ ] Comprendre la gestion des risques
- [ ] Explorer les rapports avancés
- [ ] Rejoindre la communauté des traders
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
