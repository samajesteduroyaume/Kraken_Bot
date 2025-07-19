# ❓ Foire Aux Questions (FAQ)

## Problèmes d'Installation

### Q1 : J'ai une erreur de dépendances manquantes
**R :** Assurez-vous d'avoir installé toutes les dépendances :
```bash
pip install -r requirements.txt
```

### Q2 : Python 3.12 n'est pas reconnu
**R :** Vérifiez votre version de Python :
```bash
python --version
```
Si nécessaire, installez Python 3.12+ et créez un nouvel environnement virtuel.

## Problèmes de Configuration

### Q3 : Où trouver mes clés API Kraken ?
**R :** Connectez-vous à votre compte Kraken, allez dans Paramètres > API et créez une nouvelle clé. Consultez le [guide d'authentification](getting_started/api_authentication.md) pour plus de détails.

### Q4 : Comment configurer la base de données ?
**R :** Assurez-vous que PostgreSQL est installé et que vous avez créé une base de données. Puis exécutez :
```bash
python scripts/init_db.py
```

## Problèmes d'Exécution

### Q5 : Le bot ne passe pas d'ordres
**Vérifiez :**
1. Le mode simulation est désactivé dans `.env`
2. Les clés API sont correctement configurées
3. Votre compte a suffisamment de fonds
4. La paire de trading est activée sur Kraken

### Q6 : Comment voir les logs ?
**R :** Les logs sont disponibles dans le dossier `logs/` :
```bash
tail -f logs/trading_bot.log
```

## Problèmes de Performance

### Q7 : Le bot est lent
**Solutions possibles :**
- Augmentez l'intervalle de trading
- Réduisez le nombre de paires tradées
- Vérifiez votre connexion Internet
- Optimisez votre stratégie

### Q8 : Comment réduire l'utilisation de la RAM ?
**Conseils :**
- Réduisez la taille du cache
- Limitez l'historique des données chargées
- Désactivez les indicateurs non utilisés

## Questions Courantes

### Q9 : Puis-je utiliser plusieurs stratégies en même temps ?
**R :** Oui, vous pouvez configurer plusieurs stratégies dans `config/config.yaml`.

### Q10 : Comment sauvegarder ma configuration ?
**R :** Tous les fichiers de configuration sont dans le dossier `config/`. Sauvegardez ce dossier.

## Problèmes Techniques

### Q11 : Erreur de connexion à la base de données
**Vérifiez :**
1. Que PostgreSQL est en cours d'exécution
2. Les identifiants dans `.env` sont corrects
3. Que l'utilisateur a les droits nécessaires

### Q12 : Comment mettre à jour le bot ?
```bash
git pull origin main
pip install -r requirements.txt
python scripts/init_db.py  # Si des migrations sont nécessaires
```

## Support

Pour toute autre question, ouvrez une issue sur [GitHub](https://github.com/yourusername/Kraken_Bot/issues).

## Liens Utiles

- [Guide d'installation](getting_started/installation.md)
- [Configuration de l'API](getting_started/api_authentication.md)
- [Guide des stratégies](user_guide/strategies.md)
