# 🔑 Authentification avec l'API Kraken

Ce guide explique comment configurer l'authentification avec l'API Kraken pour utiliser le bot de trading.

## 1. Création des Clés API Kraken

1. Connectez-vous à votre compte Kraken
2. Allez dans "Paramètres" > "API"
3. Cliquez sur "Ajouter une clé"
4. Configurez les permissions nécessaires :
   - **Query Funds** : Lecture seule des soldes
   - **Create and modify orders** : Création et modification d'ordres
   - **Query open orders and trades** : Consultation des ordres ouverts
   - **Access User Data** : Accès aux données utilisateur
   - **Query closed orders and trades** : Historique des trades
5. Cliquez sur "Générer la clé"
6. **Important** : Copiez immédiatement votre clé API et votre clé secrète (cette dernière ne sera affichée qu'une seule fois)

## 2. Configuration des Variables d'Environnement

Créez ou modifiez le fichier `.env` à la racine du projet :

```env
# Clés API Kraken
KRAKEN_API_KEY=votre_cle_api
KRAKEN_SECRET=votre_cle_secrete

# Paramètres de sécurité (optionnels)
KRAKEN_API_RATE_LIMIT=3  # Requêtes par seconde
KRAKEN_API_TIMEOUT=30     # Timeout en secondes
```

## 3. Vérification de la Connexion

Utilisez le script de test pour vérifier que l'authentification fonctionne :

```bash
python test_connection.py
```

## 4. Bonnes Pratiques de Sécurité

1. **Ne partagez jamais vos clés API**
2. **Utilisez des permissions minimales** nécessaires
3. **Limitez les adresses IP** qui peuvent utiliser la clé API
4. **Activez l'authentification à deux facteurs** (2FA) sur votre compte
5. **Régénérez régulièrement** vos clés API
6. **Ne versionnez jamais** vos clés API dans le dépôt Git

## 5. Dépannage

### Problème : Erreur d'authentification
- Vérifiez que les clés sont correctement copiées
- Vérifiez les permissions de la clé API
- Vérifiez que l'horloge de votre système est synchronisée (NTP)

### Problème : Limite de débit dépassée
- Réduisez la fréquence des appels API
- Augmentez la valeur de `KRAKEN_API_RATE_LIMIT` si nécessaire
- Implémentez un système de file d'attente

## 6. Configuration Avancée

### Utilisation d'un Proxy

Si vous avez besoin d'utiliser un proxy :

```env
# Configuration du proxy
HTTP_PROXY=http://user:password@proxy:port
HTTPS_PROXY=http://user:password@proxy:port
```

### Journalisation des Appels API

Activez la journalisation détaillée dans `config/logging.yaml` :

```yaml
loggers:
  krakenapi:
    level: DEBUG
    handlers: [console, file]
    propagate: no
```

## 7. Révoquer une Clé API

Si une clé est compromise ou n'est plus nécessaire :
1. Allez dans "Paramètres" > "API"
2. Trouvez la clé concernée
3. Cliquez sur "Révoquer"
4. Générez une nouvelle clé si nécessaire

## Prochaines Étapes

- [Configuration initiale](configuration.md)
- [Guide d'utilisation rapide](../user_guide/overview.md)
- [Création de stratégies personnalisées](../developer_guide/creating_strategies.md)
