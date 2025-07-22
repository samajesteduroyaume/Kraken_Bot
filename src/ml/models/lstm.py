"""
Implémentation du modèle LSTM pour le trading.
"""
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from torch.utils.data import DataLoader, Dataset
from ..core.base_model import BaseModel

# Configuration du logger
logger = logging.getLogger(__name__)


class LSTMDataset(Dataset):
    """Classe Dataset pour les données séquentielles LSTM."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Initialise le dataset LSTM.
        
        Args:
            X: Données d'entrée de forme (n_samples, seq_len, n_features)
            y: Étiquettes cibles (optionnel)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
        return (self.X[idx],)


class LSTMModel(nn.Module):
    """Modèle LSTM pour la prédiction de séries temporelles."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 50, 
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.2):
        """Initialise le modèle LSTM.
        
        Args:
            input_dim: Dimension des caractéristiques d'entrée (int)
            hidden_dim: Dimension de la couche cachée LSTM (int)
            num_layers: Nombre de couches LSTM (int)
            num_classes: Nombre de classes de sortie (int)
            dropout: Taux de dropout (float)
        """
        super().__init__()
        
        # Validation des types
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim doit être un entier positif, reçu: {input_dim} ({type(input_dim)})")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim doit être un entier positif, reçu: {hidden_dim} ({type(hidden_dim)})")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers doit être un entier positif, reçu: {num_layers} ({type(num_layers)})")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes doit être un entier positif, reçu: {num_classes} ({type(num_classes)})")
            
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Couches du modèle
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant du modèle."""
        logger.debug(f"Entrée forward - x shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
        logger.debug(f"hidden_dim: {self.hidden_dim}, num_layers: {self.num_layers}")
        
        # Vérification des dimensions d'entrée
        if x.dim() != 3:
            raise ValueError(f"L'entrée doit avoir 3 dimensions (batch, seq_len, input_size), reçu: {x.shape}")
            
        batch_size = x.size(0)
        seq_len = x.size(1)
        input_size = x.size(2)
        
        logger.debug(f"batch_size: {batch_size}, seq_len: {seq_len}, input_size: {input_size}")
        
        # Initialisation des états cachés
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        logger.debug(f"h0 shape: {h0.shape}, c0 shape: {c0.shape}")
        
        try:
            out, _ = self.lstm(x, (h0, c0))
            logger.debug(f"Sortie LSTM shape: {out.shape if out is not None else 'None'}")
        except Exception as e:
            logger.error(f"Erreur dans la passe avant LSTM: {str(e)}")
            raise
        out = self.dropout(out[:, -1, :])
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMPredictor(BaseModel):
    """Wrapper pour le modèle LSTM compatible avec l'interface BaseModel."""
    
    def _initialize_model(self) -> None:
        """Initialise le modèle LSTM avec les paramètres fournis."""
        # Extraction et validation des paramètres
        model_params = {
            'input_dim': int(self.model_params.get('input_dim', 10)),
            'hidden_dim': int(self.model_params.get('hidden_dim', 50)),
            'num_layers': int(self.model_params.get('num_layers', 2)),
            'num_classes': int(self.model_params.get('num_classes', 3)),
            'dropout': float(self.model_params.get('dropout', 0.2))
        }
        
        # Paramètres d'entraînement
        self.learning_rate = float(self.model_params.get('learning_rate', 0.001))
        self.batch_size = int(self.model_params.get('batch_size', 32))
        self.epochs = int(self.model_params.get('epochs', 50))
        
        # Détermination du périphérique (CPU/GPU)
        self.device = torch.device(
            self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        logger.info(f"Initialisation du modèle LSTM avec les paramètres: {model_params}")
        logger.info(f"Périphérique: {self.device}, Taux d'apprentissage: {self.learning_rate}")
        
        # Création du modèle
        logger.info(f"Création du modèle LSTM avec les paramètres: {model_params}")
        logger.info(f"Type de input_dim: {type(model_params['input_dim'])}, Valeur: {model_params['input_dim']}")
        self.model = LSTMModel(**model_params).to(self.device)
        logger.info(f"Modèle LSTM créé avec succès sur {self.device}")
        
        # Initialisation de l'optimiseur et de la fonction de perte
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Historique des métriques
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Entraîne le modèle LSTM.
        
        Args:
            X: Données d'entrée de forme (n_samples, seq_len, n_features)
            y: Étiquettes cibles de forme (n_samples,)
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        # Vérification des dimensions
        if len(X.shape) != 3:
            raise ValueError(
                f"X doit avoir 3 dimensions (batch, seq_len, n_features), reçu: {X.shape}"
            )
            
        # Vérification de la cohérence des données
        if X.shape[0] != len(y):
            raise ValueError(
                f"Le nombre d'échantillons dans X ({X.shape[0]}) ne correspond pas "
                f"au nombre d'étiquettes ({len(y)})"
            )
            
        logger.info(f"Début de l'entraînement sur {len(X)} échantillons")
        
        # Création des datasets et dataloaders
        dataset = LSTMDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if str(self.device) == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size * 2,  # Plus grand batch pour la validation
            shuffle=False,
            num_workers=2,
            pin_memory=True if str(self.device) == 'cuda' else False
        )
        
        # Boucle d'entraînement
        best_val_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = 5
        
        for epoch in range(self.epochs):
            # Phase d'entraînement
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass et optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistiques
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Calcul des métriques d'entraînement
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            
            # Phase de validation
            val_loss, val_acc = self._evaluate(val_loader)
            
            # Mise à jour de l'historique
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Log des métriques
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Sauvegarde du meilleur modèle
                if 'model_save_path' in kwargs:
                    self.save(kwargs['model_save_path'])
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    logger.info(f"Early stopping après {epoch+1} époques")
                    break
        
        # Chargement du meilleur modèle
        if 'model_save_path' in kwargs and os.path.exists(kwargs['model_save_path']):
            self.load(kwargs['model_save_path'])
        
        # Retour des métriques finales
        return {
            'train_loss': self.training_history['train_loss'][-1],
            'val_loss': self.training_history['val_loss'][-1],
            'train_accuracy': self.training_history['train_acc'][-1],
            'val_accuracy': self.training_history['val_acc'][-1]
        }
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Évalue le modèle sur un DataLoader donné.
        
        Args:
            data_loader: DataLoader contenant les données de validation/test
            
        Returns:
            Tuple contenant (loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Statistiques
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions avec le modèle LSTM.
        
        Args:
            X: Données d'entrée de forme (n_samples, seq_len, n_features)
            
        Returns:
            Prédictions de classe pour chaque échantillon
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"X doit avoir 3 dimensions (batch, seq_len, n_features), reçu: {X.shape}"
            )
            
        self.model.eval()
        dataset = LSTMDataset(X)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True if str(self.device) == 'cuda' else False
        )
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, tuple):
                    # Cas avec étiquettes (entraînement/évaluation)
                    batch_X, _ = batch
                else:
                    # Cas sans étiquettes (prédiction)
                    batch_X = batch
                
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def save(self, path: str) -> None:
        """Sauvegarde le modèle LSTM.
        
        Args:
            path: Chemin du dossier où sauvegarder le modèle
        """
        import joblib
        
        # Créer le dossier s'il n'existe pas
        os.makedirs(path, exist_ok=True)
        
        # Chemin complet pour le fichier du modèle
        model_path = os.path.join(path, 'model.pt')
        
        # Sauvegarder l'état du modèle et de l'optimiseur
        # Inclure les paramètres du modèle actuel pour assurer la cohérence
        saved_params = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_params': {
                'input_dim': self.model_params.get('input_dim', 10),
                'hidden_dim': self.model_params.get('hidden_dim', 50),
                'num_layers': self.model_params.get('num_layers', 2),
                'num_classes': self.model_params.get('num_classes', 3),
                'dropout': self.model_params.get('dropout', 0.2)
            },
            'training_params': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'device': str(self.device)
            },
            'last_trained': self.last_trained
        }
        torch.save(saved_params, model_path)
        
        # Sauvegarder le scaler s'il a été ajusté
        if hasattr(self, 'scaler') and self.scaler is not None:
            scaler_path = os.path.join(path, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
        # Sauvegarder également les métadonnées dans un fichier séparé
        metadata = {
            'name': self.name,
            'model_type': 'lstm',
            'model_params': self.model_params,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'has_scaler': hasattr(self, 'scaler') and self.scaler is not None
        }
        metadata_path = os.path.join(path, 'metadata.joblib')
        import joblib
        joblib.dump(metadata, metadata_path)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMPredictor':
        """Charge un modèle LSTM sauvegardé.
        
        Args:
            path: Chemin du dossier contenant les fichiers du modèle
            
        Returns:
            Instance de LSTMPredictor chargée
        """
        # Chemin complet pour le fichier du modèle
        model_path = os.path.join(path, 'model.pt')
        metadata_path = os.path.join(path, 'metadata.joblib')
        scaler_path = os.path.join(path, 'scaler.joblib')
        
        # Vérifier que les fichiers nécessaires existent
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier modèle introuvable: {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Fichier de métadonnées introuvable: {metadata_path}")
        
        # Charger les métadonnées
        import joblib
        metadata = joblib.load(metadata_path)
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Récupérer les paramètres du modèle sauvegardé
        saved_model_params = checkpoint.get('model_params')
        saved_training_params = checkpoint.get('training_params', {})
        
        if not saved_model_params:
            raise ValueError("Aucun paramètre de modèle trouvé dans le fichier de sauvegarde")
        
        # Créer une instance du modèle avec les paramètres sauvegardés
        # Utiliser directement les paramètres sauvegardés sans valeurs par défaut
        model = cls(
            name=metadata.get('name', os.path.basename(path)),
            model_params=saved_model_params  # Utiliser directement les paramètres sauvegardés
        )
        
        # Mettre à jour les paramètres d'entraînement
        if saved_training_params:
            model.learning_rate = saved_training_params.get('learning_rate', 0.001)
            model.batch_size = saved_training_params.get('batch_size', 32)
            model.epochs = saved_training_params.get('epochs', 50)
            
            # Configurer le périphérique
            device_str = saved_training_params.get('device', 'cpu')
            if 'cuda' in device_str and not torch.cuda.is_available():
                logger.warning("CUDA n'est pas disponible, utilisation du CPU à la place")
                device_str = 'cpu'
            model.device = torch.device(device_str)
        
        # Charger l'état du modèle et de l'optimiseur
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.last_trained = checkpoint.get('last_trained')
        
        # Charger le scaler s'il existe
        if os.path.exists(scaler_path):
            model.scaler = joblib.load(scaler_path)
        
        # Déplacer le modèle sur le bon périphérique
        model.model = model.model.to(model.device)
        
        return model
