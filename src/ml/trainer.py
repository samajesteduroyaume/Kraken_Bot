"""
Module trainer - Contient les fonctions pour l'entraînement des modèles de machine learning.
"""
import os
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_dim).to(
            x.device)
        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_dim).to(
            x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim).to(device),
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim).to(device))


def create_lstm_model(input_dim: int, num_classes: int = 3) -> LSTMModel:
    """
    Crée un modèle LSTM pour la prédiction du marché.

    Args:
        input_dim (int): Nombre de caractéristiques d'entrée
        num_classes (int): Nombre de classes de sortie (3 pour [vendre, maintenir, acheter])

    Returns:
        LSTMModel: Modèle LSTM PyTorch
    """
    return LSTMModel(input_dim, num_classes=num_classes)


def prepare_data(df,
                 feature_columns: list,
                 target_column: str,
                 sequence_length: int = 60) -> tuple[np.ndarray,
                                                     np.ndarray,
                                                     StandardScaler]:
    """
    Prépare les données pour l'entraînement du modèle LSTM.

    Args:
        df (pd.DataFrame): DataFrame contenant les données
        feature_columns (list): Liste des colonnes à utiliser comme caractéristiques
        target_column (str): Nom de la colonne cible
        sequence_length (int): Nombre de pas de temps pour chaque séquence

    Returns:
        tuple: (X, y) - Données d'entrée et étiquettes préparées
    """
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    # Création des séquences
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_features[i - sequence_length:i])
        y.append(df[target_column].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def train_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: tuple,
        model_path: str = 'models/market_predictor.pth') -> LSTMModel:
    """
    Entraîne un modèle LSTM sur les données fournies.

    Args:
        X_train (np.array): Données d'entraînement
        y_train (np.array): Étiquettes d'entraînement
        X_val (np.array): Données de validation
        y_val (np.array): Étiquettes de validation
        input_shape (tuple): Forme des données d'entrée
        model_path (str): Chemin pour sauvegarder le modèle entraîné

    Returns:
        LSTMModel: Modèle entraîné
    """
    # Créer le répertoire des modèles s'il n'existe pas
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Préparer les données pour PyTorch
    train_dataset = MarketDataset(X_train, y_train)
    val_dataset = MarketDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Créer le modèle
    model = create_lstm_model(input_shape[-1])
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(model.device)
                y_batch = y_batch.to(model.device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100. * correct / total

        logger.info(
            f'Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {accuracy:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break

    # Charger le meilleur modèle
    model.load_state_dict(torch.load(model_path))
    logger.info(f"Modèle entraîné et sauvegardé dans {model_path}")
    return model


def evaluate_model(
        model: LSTMModel,
        X_test: np.ndarray,
        y_test: np.ndarray) -> dict:
    """
    Évalue les performances du modèle sur un ensemble de test.

    Args:
        model (LSTMModel): Modèle à évaluer
        X_test (np.array): Données de test
        y_test (np.array): Étiquettes de test

    Returns:
        dict: Métriques d'évaluation
    """
    test_dataset = MarketDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(y_batch.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    metrics = {
        'loss': test_loss,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }

    return metrics
