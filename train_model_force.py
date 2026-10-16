import sys
import os
import torch
import pickle
sys.path.append(os.path.abspath('.'))
os.environ['LEAGUE_CONFIG'] = 'mafia'
from src.preprocessing import prepare_ml_dataset, split_and_scale_data
from src.model import FantasyDataset, FantasyNN, train_model
from torch.utils.data import DataLoader

print("Preparing ML Dataset...")
df = prepare_ml_dataset()
X_train, y_train_adp, y_train_pts, X_val, y_val_adp, y_val_pts, scaler, feature_cols = split_and_scale_data(df)

print(f"Features: {feature_cols}")

train_dataset_adp = FantasyDataset(X_train, y_train_adp)
val_dataset_adp = FantasyDataset(X_val, y_val_adp)
train_loader_adp = DataLoader(train_dataset_adp, batch_size=64, shuffle=True)
val_loader_adp = DataLoader(val_dataset_adp, batch_size=64, shuffle=False)

train_dataset_pts = FantasyDataset(X_train, y_train_pts)
val_dataset_pts = FantasyDataset(X_val, y_val_pts)
train_loader_pts = DataLoader(train_dataset_pts, batch_size=64, shuffle=True)
val_loader_pts = DataLoader(val_dataset_pts, batch_size=64, shuffle=False)


input_dim = X_train.shape[1]
print(f"Input Dimension: {input_dim}")

model_adp = FantasyNN(input_dim)
model_pts = FantasyNN(input_dim)

print("Training ADP Model...")
train_model(model_adp, train_loader_adp, val_loader_adp, epochs=150)

print("Training PTS Model...")
train_model(model_pts, train_loader_pts, val_loader_pts, epochs=150)

torch.save(model_adp.state_dict(), 'models/model_adp.pth')
torch.save(model_pts.state_dict(), 'models/model_pts.pth')
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
    
print("Successfully retrained and saved models!")
