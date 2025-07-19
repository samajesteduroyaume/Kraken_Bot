# 🛡️ Gestion des Risques

La gestion des risques est essentielle pour un trading réussi. Kraken_Bot propose plusieurs outils pour vous aider à gérer et à atténuer les risques.

## Paramètres de Gestion des Risques

### 1. Stop-Loss et Take-Profit

Configurez des niveaux de sortie automatiques pour chaque trade :

```yaml
risk_management:
  # Stop-loss en pourcentage (négatif)
  stop_loss: -0.05  # -5%
  
  # Take-profit en pourcentage (positif)
  take_profit: 0.10  # +10%
  
  # Trailing stop (optionnel)
  trailing_stop: true
  trailing_stop_distance: 0.02  # 2%
```

### 2. Gestion de la Taille des Positions

Contrôlez la taille de vos positions pour gérer votre exposition au risque :

```yaml
position_sizing:
  # Montant maximum à risquer par trade (en devise de base)
  max_risk_per_trade: 100  # EUR, USD, etc.
  
  # Pourcentage maximum du capital à risquer par trade
  max_risk_percent: 1.0  # 1% du capital total
  
  # Taille de position fixe (prioritaire si défini)
  fixed_position_size: null  # Laissez à null pour utiliser le calcul dynamique
```

### 3. Limites Globales

Définissez des limites pour l'ensemble de votre portefeuille :

```yaml
portfolio_limits:
  # Nombre maximum de positions ouvertes simultanément
  max_open_trades: 5
  
  # Exposition maximale en pourcentage du capital
  max_exposure: 20.0  # 20% du capital total
  
  # Limite de perte quotidienne
  daily_loss_limit: -5.0  # -5% du capital
  
  # Limite de perte globale
  max_drawdown: -20.0  # -20% du capital initial
```

## Stratégies de Gestion des Risques

### 1. Volatilité Adaptative

Ajuste automatiquement la taille des positions en fonction de la volatilité du marché :

```yaml
volatility_adjustment:
  enabled: true
  atr_period: 14
  atr_multiplier: 2.0
  min_position_size: 10  # Taille minimale de position
  max_position_size: 1000  # Taille maximale de position
```

### 2. Corrélation des Actifs

Évitez une exposition excessive à des actifs corrélés :

```yaml
correlation_control:
  enabled: true
  correlation_threshold: 0.7  # 0.0 à 1.0
  lookback_period: 30  # Jours
  max_correlated_pairs: 2
```

### 3. Gestion des Nouvelles du Marché

Réduisez automatiquement l'exposition pendant les périodes de forte volatilité :

```yaml
news_aware_trading:
  enabled: true
  reduce_exposure: true
  max_exposure_during_high_vol: 0.5  # 50% de l'exposition normale
  volatility_threshold: 1.5  # Multiplicateur de la volatilité moyenne
```

## Surveillance et Alertes

### Alertes de Risque

Recevez des notifications lorsque des seuils de risque sont atteints :

```yaml
alerts:
  # Alertes par email
  email:
    enabled: true
    recipients: ["votre@email.com"]
    
    # Seuils d'alerte
    max_drawdown_alert: -10.0  # %
    daily_loss_alert: -5.0     # %
    position_size_alert: 10    # % du capital
```

### Tableau de Bord de Risque

Accédez à un aperçu de votre exposition au risque :

```bash
python manage_config.py --risk-dashboard
```

## Meilleures Pratiques

1. **Commencez petit** : Testez avec de petites positions
2. **Diversifiez** : Ne mettez pas tous vos œufs dans le même panier
3. **Utilisez des stops** : Toujours définir des stops-loss
4. **Surveillez** : Vérifiez régulièrement votre exposition
5. **Restez informé** : Tenez-vous au courant des actualités du marché

## Dépannage

- **Stops non déclenchés** : Vérifiez les paramètres de votre courtier
- **Slippage important** : Évitez les périodes de forte volatilité
- **Exposition trop élevée** : Ajustez les limites de position

## Prochaines Étapes

- [Stratégies de trading](strategies.md)
- [Configuration avancée](../getting_started/configuration.md)
- [API de trading](../api_reference/overview.md)
