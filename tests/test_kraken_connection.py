#!/usr/bin/env python3
"""
Script de test pour vérifier la connexion à l'API Kraken.
"""
import os
import asyncio
import aiohttp
import base64
import hashlib
import hmac
import json
import time
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
KRAKEN_API_URL = 'https://api.kraken.com'
API_VERSION = '0'

def get_kraken_signature(urlpath, data, secret):
    """Génère la signature pour l'API Kraken."""
    postdata = ''
    if data:
        postdata = '&'.join([f"{k}={v}" for k, v in data.items()])
    
    # Créer le message à signer
    message = urlpath.encode() + hashlib.sha256(
        (str(data.get('nonce', '')) + postdata).encode()
    ).digest()
    
    # Décoder la clé secrète
    decoded_secret = base64.b64decode(secret)
    
    # Créer la signature HMAC-SHA512
    hmac_sha512 = hmac.new(decoded_secret, message, hashlib.sha512)
    signature = base64.b64encode(hmac_sha512.digest())
    
    return signature.decode()

async def test_public_endpoint():
    """Teste un endpoint public de l'API Kraken."""
    endpoint = f"{KRAKEN_API_URL}/{API_VERSION}/public/Time"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'error' in data and not data['error']:
                        print("✅ Connexion à l'API publique réussie!")
                        print(f"Heure du serveur: {data.get('result', {}).get('rfc1123', 'Inconnue')}")
                        return True
                    else:
                        print(f"❌ Erreur de l'API: {data.get('error')}")
                else:
                    print(f"❌ Erreur HTTP {response.status}")
        except Exception as e:
            print(f"❌ Erreur de connexion: {str(e)}")
    
    return False

async def test_private_endpoint():
    """Teste un endpoint privé de l'API Kraken."""
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        print("❌ Clés API manquantes dans le fichier .env")
        return False
    
    endpoint_path = f"/{API_VERSION}/private/Balance"
    url = f"{KRAKEN_API_URL}{endpoint_path}"
    
    # Préparer les données de la requête
    nonce = str(int(time.time() * 1000))
    data = {
        'nonce': nonce
    }
    
    # Générer la signature
    signature = get_kraken_signature(endpoint_path, data, KRAKEN_API_SECRET)
    
    # Préparer les en-têtes
    headers = {
        'API-Key': KRAKEN_API_KEY,
        'API-Sign': signature,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, data=data) as response:
                response_data = await response.json()
                
                if response.status == 200 and 'error' in response_data and not response_data['error']:
                    print("✅ Connexion à l'API privée réussie!")
                    print(f"Solde disponible: {len(response_data.get('result', {}))} actifs")
                    return True
                else:
                    print(f"❌ Erreur de l'API: {response_data.get('error')}")
                    return False
        except Exception as e:
            print(f"❌ Erreur de connexion: {str(e)}")
            return False

async def test_asset_pairs():
    """Teste la récupération des paires d'actifs."""
    endpoint = f"{KRAKEN_API_URL}/{API_VERSION}/public/AssetPairs"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'error' in data and not data['error']:
                        pairs = data.get('result', {})
                        print(f"✅ {len(pairs)} paires d'actifs récupérées avec succès!")
                        print("\nExemples de paires:")
                        for i, pair in enumerate(list(pairs.items())[:5], 1):
                            print(f"{i}. {pair[0]}: {pair[1].get('wsname', 'N/A')}")
                        return True
                    else:
                        print(f"❌ Erreur de l'API: {data.get('error')}")
                else:
                    print(f"❌ Erreur HTTP {response.status}")
        except Exception as e:
            print(f"❌ Erreur de connexion: {str(e)}")
    
    return False

async def main():
    print("=== Test de connexion à l'API Kraken ===\n")
    
    # Tester la connexion à l'API publique
    print("1. Test de connexion à l'API publique...")
    public_success = await test_public_endpoint()
    
    # Tester la connexion à l'API privée
    print("\n2. Test de connexion à l'API privée...")
    private_success = await test_private_endpoint()
    
    # Tester la récupération des paires d'actifs
    print("\n3. Test de récupération des paires d'actifs...")
    pairs_success = await test_asset_pairs()
    
    # Afficher un résumé
    print("\n=== Résumé des tests ===")
    print(f"API publique: {'✅ Réussi' if public_success else '❌ Échec'}")
    print(f"API privée: {'✅ Réussi' if private_success else '❌ Échec'}")
    print(f"Récupération des paires: {'✅ Réussi' if pairs_success else '❌ Échec'}")

if __name__ == "__main__":
    asyncio.run(main())
