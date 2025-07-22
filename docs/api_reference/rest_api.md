# 📚 Référence de l'API REST

L'API REST de Kraken_Bot permet d'interagir avec la plateforme de trading de manière programmatique. Ce document détaille tous les endpoints disponibles, les paramètres et les réponses.

## 🔑 Authentification

Toutes les requêtes nécessitent une clé API valide dans l'en-tête `X-API-KEY`.

```http
GET /api/v1/account
X-API-KEY: votre_cle_api_secrete
```

## 📦 Format des Réponses

Toutes les réponses sont au format JSON et incluent :

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2025-03-20T14:30:00Z"
}
```

## 📊 Endpoints Principaux

### 1. Compte et Portefeuille

#### Obtenir le solde du compte
```http
GET /api/v1/account/balance
```

**Réponse** :
```json
{
  "success": true,
  "data": {
    "total_balance": 10000.50,
    "available_balance": 8000.25,
    "currencies": [
      {
        "currency": "BTC",
        "balance": 1.5,
        "available": 1.2,
        "hold": 0.3
      }
    ]
  }
}
```

### 2. Marchés et Données

#### Obtenir les paires de trading
```http
GET /api/v1/markets
```

**Paramètres** :
- `type` : `spot`, `futures`, `all` (par défaut)
- `quote` : Filtre par devise de cotation (ex: `EUR`)

### 3. Trading

#### Passer un ordre
```http
POST /api/v1/orders
```

**Corps de la requête** :
```json
{
  "symbol": "BTC/EUR",
  "side": "buy",
  "type": "limit",
  "amount": 0.1,
  "price": 50000,
  "params": {
    "timeInForce": "GTC",
    "postOnly": true
  }
}
```

### 4. Stratégies

#### Lister les stratégies actives
```http
GET /api/v1/strategies
```

#### Démarrer une stratégie
```http
POST /api/v1/strategies/start
```

**Corps de la requête** :
```json
{
  "strategy": "mean_reversion",
  "symbol": "BTC/EUR",
  "params": {
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0
  }
}
```

## 🔄 Webhooks

### Configuration des webhooks
```http
POST /api/v1/webhooks
```

**Corps de la requête** :
```json
{
  "url": "https://votre-webhook.com/endpoint",
  "events": ["order.filled", "position.closed"],
  "secret": "votre_secret_pour_la_validation"
}
```

## 📈 Données Historiques

### Obtenir les bougies OHLCV
```http
GET /api/v1/ohlcv/BTC/EUR
```

**Paramètres** :
- `timeframe` : `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- `limit` : Nombre de bougies à retourner (max 1000)
- `start_time` : Timestamp de début
- `end_time` : Timestamp de fin

## 🔐 Sécurité

### Rotation des clés API
```http
POST /api/v1/account/rotate-keys
```

## 📚 Exemples de Code

### Python
```python
import requests
import hmac
import hashlib
import time

def get_balance(api_key, api_secret):
    url = "https://api.krakenbot.com/api/v1/account/balance"
    timestamp = str(int(time.time() * 1000))
    
    signature = hmac.new(
        api_secret.encode(),
        f"{timestamp}GET/api/v1/account/balance".encode(),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        'X-API-KEY': api_key,
        'X-API-TIMESTAMP': timestamp,
        'X-API-SIGNATURE': signature
    }
    
    response = requests.get(url, headers=headers)
    return response.json()
```

### JavaScript (Node.js)
```javascript
const crypto = require('crypto');

async function getBalance(apiKey, apiSecret) {
  const timestamp = Date.now().toString();
  const method = 'GET';
  const path = '/api/v1/account/balance';
  
  const signature = crypto
    .createHmac('sha256', apiSecret)
    .update(`${timestamp}${method}${path}`)
    .digest('hex');
    
  const response = await fetch('https://api.krakenbot.com' + path, {
    headers: {
      'X-API-KEY': apiKey,
      'X-API-TIMESTAMP': timestamp,
      'X-API-SIGNATURE': signature
    }
  });
  
  return await response.json();
}
```

## 📊 Codes d'Erreur

| Code | Description | Solution |
|------|-------------|-----------|
| 400 | Mauvaise requête | Vérifiez les paramètres |
| 401 | Non autorisé | Vérifiez votre clé API |
| 403 | Interdit | Vérifiez les permissions |
| 404 | Non trouvé | L'endpoint n'existe pas |
| 429 | Trop de requêtes | Réduisez la fréquence |
| 500 | Erreur serveur | Réessayez plus tard |

## 📞 Support

Pour toute question technique concernant l'API :
- Email : api@krakenbot.com
- Documentation en ligne : https://docs.krakenbot.com/api
- Statut de l'API : https://status.krakenbot.com
