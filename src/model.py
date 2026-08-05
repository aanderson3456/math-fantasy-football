import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class FantasyDataset(Dataset):
    """Custom PyTorch Dataset for loading scaled fantasy football features and target ADPs."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # shape: (N, 1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FantasyNN(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Neural Network for predicting player draft value (ADP).
    Uses Batch Normalization, Dropout, and GELU activations.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_prob=0.15):
        super(FantasyNN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h_dim
            
        # Output layer (predicting a continuous value: ADP)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, epochs=150, lr=0.002, weight_decay=1e-4, patience=15):
    """
    Trains the PyTorch model with early stopping on validation loss.
    Returns:
        model: Trained model with the best validation weights restored.
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Learning rate schedule step
        scheduler.step(epoch_val_loss)
        
        # Print progress occasionally
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
            
        # Early Stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break
                
    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, train_losses, val_losses

def evaluate_model(model, X, y_true):
    """Calculates regression metrics (MAE, RMSE, R2) for the PyTorch model."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        predictions = model(inputs).numpy().flatten()
        
    # Clamp predictions to a minimum ADP of 1.0 (no one can be drafted before pick 1)
    predictions = np.clip(predictions, 1.0, None)
    
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)
    
    return mae, rmse, r2, predictions
