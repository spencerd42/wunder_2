import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataset import SubsequenceDataset, split_and_scale
from tqdm import tqdm
import os
from pandas import DataFrame
from sklearn.metrics import r2_score
import pickle

from architecture import TimeSeriesLSTM, Small_LSTM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_dataloaders(data: DataFrame):
    # Split and scale the data
    train_data, test_data, scaler = split_and_scale(data)

    # Convert back to DataFrames for the dataset class
    train_data_df = pd.DataFrame(train_data)
    test_data_df = pd.DataFrame(test_data)

    batch_size = 256
    window_size = 100

    # Create dataset instances
    train_dataset = SubsequenceDataset(train_data_df, window_size=window_size)
    test_dataset = SubsequenceDataset(test_data_df, window_size=window_size)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

def train(input_size, train_loader, test_loader):

    # Hyperparameters
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 8

    # Initialize model
    model = TimeSeriesLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=input_size
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                
                output = model(x)
                loss = criterion(output, y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Step the scheduler
        scheduler.step(avg_test_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Test Loss: {avg_test_loss:.6f}")

    # Save model checkpoint
    torch.save(model.state_dict(), 'model_checkpoint.pth')
    print("\nModel saved to 'model_checkpoint.pth'")

    # Calculate final metrics
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    
    return model

# ============================================================================
# STEP 7: MAKE PREDICTIONS (EXAMPLE)
# ============================================================================

def make_predictions(model, test_loader, device, scaler):
    """Generate predictions on test set"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            output = model(x)
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform if needed
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f"Test RMSE: {rmse:.6f}")
    print(r2_score(actuals, predictions))
    
    return predictions, actuals

# predictions, actuals = make_predictions(model, test_loader, device, scaler)

if __name__ == "__main__":
    data = pd.read_parquet(f'{CURRENT_DIR}/data/train.parquet')
    train_loader, test_loader, scaler = make_dataloaders(data)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # model = train(32, train_loader, test_loader)

    # Initialize model architecture (with correct parameters)
    model = TimeSeriesLSTM(input_size=32, hidden_size=64, num_layers=2, output_size=32)

    # Load checkpoint
    state_dict = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Switch to eval mode
    model.eval()
    
    make_predictions(model, test_loader, torch.device('cpu'), scaler)