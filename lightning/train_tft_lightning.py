import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import the Lightning model and dataset
from tft_lightning_model import TemporalFusionTransformerLightning, MetricsCallback
from dataset import make_dataloaders
from config_lightning import LightningTrainingConfig


def train_tft_lightning(config: LightningTrainingConfig):
    """
    Train TFT model using PyTorch Lightning
    
    Args:
        config: LightningTrainingConfig object with training parameters
    """
    
    # Set random seed for reproducibility
    pl.seed_everything(config.random_seed)
    
    # Setup device
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
    elif torch.mps.is_available():
        accelerator = 'mps'
        devices = 1
    else:
        accelerator = 'cpu'
        devices = 1
    
    print("="*70)
    print(f"PYTORCH LIGHTNING TRAINING - TFT MODEL")
    print("="*70)
    print(f"Accelerator: {accelerator}")
    print(f"Config: {config.to_dict()}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data = pd.read_parquet(config.data_path)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = make_dataloaders(
        data, config.window_size, config.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing TFT model...")
    model = TemporalFusionTransformerLightning(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_lstm_layers=config.num_lstm_layers,
        dropout=config.dropout,
        output_size=config.input_size,
        learning_rate=config.learning_rate,
        lr_scheduler_factor=config.lr_scheduler_factor,
        lr_scheduler_patience=config.lr_scheduler_patience
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='tft-{epoch:02d}-{val_loss:.6f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=config.early_stopping_min_delta,
        patience=config.early_stopping_patience,
        verbose=True,
        mode='min'
    )
     
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    metrics_callback = MetricsCallback()
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config.checkpoint_dir,
        name='tft_logs',
        version=None
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            metrics_callback
        ],
        logger=logger,
        gradient_clip_val=1.0,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision='16-mixed' if accelerator == 'gpu' else '32'
    )
    
    # Train the model
    print("\nStarting training...")
    print("="*70)
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model for testing
    print("\n" + "="*70)
    print("Loading best model for testing...")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    
    if best_model_path:
        model = TemporalFusionTransformerLightning.load_from_checkpoint(best_model_path)
    
    # Test the model
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, test_loader)
    
    # Print final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model checkpoint: {best_model_path}")
    print(f"Test loss: {test_results[0]['test_loss']:.6f}")
    
    # Save training configuration
    config_path = checkpoint_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    return model, trainer


def predict_with_model(model_checkpoint_path, test_loader, device='cpu'):
    """
    Load a trained model and generate predictions
    
    Args:
        model_checkpoint_path: Path to the checkpoint file
        test_loader: DataLoader for test data
        device: Device to run inference on
        
    Returns:
        predictions: numpy array of predictions
        actuals: numpy array of actual values
    """
    # Load model from checkpoint
    model = TemporalFusionTransformerLightning.load_from_checkpoint(model_checkpoint_path)
    model.eval()
    model.to(device)
    
    predictions = []
    actuals = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            output = model(x)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = r2_score(actuals, predictions)
    
    print("\n" + "="*70)
    print("PREDICTION METRICS")
    print("="*70)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    return predictions, actuals


def main():
    """Main training script"""
    
    # Create configuration
    config = LightningTrainingConfig()
    
    # Train model
    model, trainer = train_tft_lightning(config)
    
    print("\n" + "="*70)
    print("To view training logs, run:")
    print(f"tensorboard --logdir {config.checkpoint_dir}/tft_logs")
    print("="*70)


if __name__ == "__main__":
    main()
