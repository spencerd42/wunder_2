import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tft_lightning_model import TemporalFusionTransformerLightning
from config_lightning import LightningTrainingConfig

def predict_with_model(model_checkpoint_path, test_loader, device='cpu'):
    """
    Load a trained model and generate predictions with evaluation metrics.

    Args:
        model_checkpoint_path (str): Path to the PyTorch Lightning checkpoint file.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device to run inference on ('cpu', 'cuda', 'mps', etc.).

    Returns:
        tuple: (predictions, actuals) numpy arrays.
    """
    # Load the model from checkpoint
    model = TemporalFusionTransformerLightning.load_from_checkpoint(model_checkpoint_path)
    model.eval()
    model.to(device)

    predictions = []
    actuals = []

    print('predicting')
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("="*60)
    print("Prediction Metrics on Test Set:")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    print("="*60)

    return predictions, actuals

# Example usage:
if __name__ == "__main__":
    from dataset import make_dataloaders
    import pandas as pd
    import os
    
    config = LightningTrainingConfig()

    data = pd.read_parquet(config.data_path)

    # Prepare dataloaders (update parameters as needed)
    _, _, test_loader = make_dataloaders(data, window_size=100, batch_size=256)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, '../checkpoints_lightning/tft-epoch=16-val_loss=0.603450.ckpt')
    print(checkpoint_path)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run prediction and evaluation
    predict_with_model(checkpoint_path, test_loader, device)
