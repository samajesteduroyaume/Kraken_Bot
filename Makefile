# Makefile pour le projet Kraken Trading Bot

# Variables
PYTHON = python3
PIP = pip3
DOCKER = docker
DOCKER_COMPOSE = docker-compose
PYTEST = pytest
COVERAGE = coverage
BLACK = black
ISORT = isort
FLAKE8 = flake8
MYPY = mypy

# Chemins
SRC_DIR = src
TESTS_DIR = tests
SCRIPTS_DIR = scripts

# Commandes principales
.PHONY: install test lint format check-style type-check build run stop clean help

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  install     Installer les dépendances du projet"
	@echo "  test        Exécuter les tests"
	@echo "  test-cov    Exécuter les tests avec couverture"
	@echo "  lint        Vérifier la qualité du code avec flake8"
	@echo "  format      Formater le code avec black et isort"
	@echo "  type-check  Vérifier les types avec mypy"
	@echo "  check-style Vérifier le style et les types (lint + type-check)"
	@echo "  build       Construire l'image Docker"
	@echo "  up          Démarrer les services avec docker-compose"
	@echo "  down        Arrêter les services avec docker-compose"
	@echo "  clean       Nettoyer les fichiers générés"

# Installation
install:
	@echo "Installation des dépendances..."
	$(PIP) install -U pip
	$(PIP) install -e .[dev]
	pre-commit install

# Tests
TEST_OPTS = -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=xml:coverage.xml

test:
	@echo "Exécution des tests..."
	$(PYTEST) $(TEST_OPTS) $(TESTS_DIR)

test-cov:
	@echo "Exécution des tests avec couverture..."
	$(PYTEST) $(TEST_OPTS) --cov-report=html:htmlcov $(TESTS_DIR)

# Vérification de la qualité du code
lint:
	@echo "Vérification avec flake8..."
	$(FLAKE8) $(SRC_DIR) $(TESTS_DIR)

format:
	@echo "Formatage avec black..."
	$(BLACK) $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	@echo "Tri des imports avec isort..."
	$(ISORT) $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

type-check:
	@echo "Vérification des types avec mypy..."
	$(MYPY) $(SRC_DIR) $(TESTS_DIR)

check-style: lint type-check

# Docker
build:
	@echo "Construction de l'image Docker..."
	$(DOCKER_COMPOSE) build

up:
	@echo "Démarrage des services avec docker-compose..."
	$(DOCKER_COMPOSE) up -d

down:
	@echo "Arrêt des services..."
	$(DOCKER_COMPOSE) down

# Nettoyage
clean:
	@echo "Nettoyage des fichiers générés..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .coverage coverage.xml htmlcov/
	rm -rf build/ dist/ *.egg-info/

# Dépendances
.DEFAULT_GOAL := help
