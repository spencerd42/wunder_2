import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from dataset import make_dataloaders
from architecture_with_tft import TimeSeriesLSTM, AttentionLSTM, TemporalFusionTransformer
from config_with_tft import TrainingConfig

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = r2_score(actuals, predictions)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating test set"):
            x = x.to(device)
            output = model(x)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = calculate_metrics(predictions, actuals)

    print(f"Test set evaluation metrics:\n"
          f"MSE: {metrics['mse']:.6f}\n"
          f"RMSE: {metrics['rmse']:.6f}\n"
          f"MAE: {metrics['mae']:.6f}\n"
          f"R^2: {metrics['r2']:.6f}\n")

    return predictions, actuals, metrics

if __name__ == "__main__":
    config = TrainingConfig('tft')

    device = torch.device('cpu')

    # Load your raw data
    data = pd.read_parquet('data/train.parquet')

    _, _, test_loader = make_dataloaders(data, config.window_size, config.batch_size)

    # Initialize model architecture (match training params)
    input_size = config.input_size
    # model = TimeSeriesLSTM(
    #     input_size=input_size,
    #     hidden_size=config.hidden_size,
    #     fc_size=config.fc_size,
    #     num_layers=config.num_layers,
    #     output_size=input_size
    # )
    # model = AttentionLSTM(
    #     input_size=input_size,
    #     hidden_size=config.hidden_size,
    #     fc_size=config.fc_size,
    #     num_layers=config.num_layers,
    #     output_size=input_size
    # )
    model = TemporalFusionTransformer(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_lstm_layers=config.num_lstm_layers,
        dropout=config.dropout,
        output_size=config.input_size
    )
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)

    # Run evaluation
    evaluate(model, test_loader, device)
