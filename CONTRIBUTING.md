# 👥 Guide de Contribution

Merci de votre intérêt pour Kraken_Bot ! Nous apprécions toutes les contributions, qu'il s'agisse de rapports de bugs, de demandes de fonctionnalités, ou de contributions de code.

## 📋 Comment Contribuer

### Rapporter un Bug
1. Vérifiez que le bug n'a pas déjà été signalé dans les [issues](https://github.com/yourusername/Kraken_Bot/issues)
2. Si ce n'est pas le cas, ouvrez une nouvelle issue avec une description claire et détaillée
3. Incluez des étapes pour reproduire le bug si possible

### Proposer une Amélioration
1. Vérifiez qu'une issue similaire n'existe pas déjà
2. Ouvrez une nouvelle issue avec le préfixe "[AMÉLIORATION]"
3. Décrivez clairement le changement proposé et son intérêt

### Soumettre une Pull Request
1. Forkez le dépôt et créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
2. Effectuez vos modifications
3. Ajoutez des tests si nécessaire
4. Assurez-vous que les tests passent
5. Soumettez une Pull Request avec une description claire

## 🛠 Environnement de Développement

### Prérequis
- Python 3.12+
- PostgreSQL
- Redis
- Git

### Configuration
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/yourusername/Kraken_Bot.git
   cd Kraken_Bot
   ```

2. Créez et activez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Linux/Mac
   # OU
   .\venv\Scripts\activate  # Sur Windows
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Configurez les variables d'environnement (voir `.env.example`)

## 📚 Documentation

La documentation est gérée avec MkDocs. Pour la prévisualiser localement :

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## ✅ Tests

Exécutez les tests avec :

```bash
# Tous les tests
pytest

# Tests avec couverture
pytest --cov=src --cov-report=html

# Un fichier de test spécifique
pytest tests/test_strategies.py -v
```

## 🧹 Vérifications avant Soumission

Avant de soumettre votre code, assurez-vous de :

1. Exécuter les tests : `pytest`
2. Vérifier la couverture de code : `pytest --cov=src`
3. Vérifier le style de code : `black . && flake8`
4. Mettre à jour la documentation si nécessaire

## 📝 Bonnes Pratiques de Développement

- Suivez le style de code existant
- Écrivez des tests pour les nouvelles fonctionnalités
- Documentez votre code avec des docstrings
- Gardez les commits atomiques et bien décrits
- Mettez à jour le CHANGELOG.md pour les changements notables

## 📜 Code de Conduite

En participant à ce projet, vous acceptez de respecter le [Code de Conduite](CODE_OF_CONDUCT.md).

## 🙏 Remerciements

Merci à tous les contributeurs qui aident à améliorer Kraken_Bot !
