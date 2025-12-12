import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.trainer import Trainer

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import the Lightning model and dataset
from tft_tpu_optimized import TemporalFusionTransformerLightning, MetricsCallback
from dataset import make_dataloaders
from config_lightning import LightningTrainingConfig



def train_tft_lightning(config: LightningTrainingConfig):
    """
    Train TFT model using PyTorch Lightning - TPU OPTIMIZED
    
    Args:
        config: LightningTrainingConfig object with training parameters
    """
    
    # Set random seed for reproducibility
    L.seed_everything(config.random_seed)
    
    # ========================================================================
    # TPU OPTIMIZATION: Use computed accelerator, don't force TPU
    # ========================================================================
    
    # Setup device - compute intelligently
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 'auto'  # Use all available GPUs
        torch.set_float32_matmul_precision('high')
        precision = '32'  # Use 32 for stability; can use '16-mixed' for speedup
    elif torch.mps.is_available():
        accelerator = 'mps'
        devices = 1
        precision = '32'
    else:
        # Only use TPU if GPU/MPS not available
        # For TPU, you should be running on Colab/GCP already
        try:
            import torch_xla
            accelerator = 'tpu'
            devices = 'auto'  # Will use available TPU cores
            precision = 'bf16-true'  # TPU prefers lower precision
        except ImportError:
            accelerator = 'cpu'
            devices = 1
            precision = '32'
    
    print("="*70)
    print(f"PYTORCH LIGHTNING TRAINING - TFT MODEL (TPU OPTIMIZED)")
    print("="*70)
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Precision: {precision}")
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
    
    # ========================================================================
    # TPU OPTIMIZATION: Ensure large batch size for XLA efficiency
    # ========================================================================
    if accelerator == 'tpu':
        print(f"\n[TPU OPTIMIZATION] Current batch size: {config.batch_size}")
        if config.batch_size < 128:
            print(f"[WARNING] Consider increasing batch_size to >=128 for TPU efficiency")
    
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
    
    # ========================================================================
    # TPU OPTIMIZATION: Trainer configuration for TPU
    # ========================================================================
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
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
        # ====================================================================
        # Additional TPU optimizations
        # ====================================================================
        enable_progress_bar=True,  # Set to False for Colab/Kaggle if verbose logs
        # Use bfloat16 on TPU for better compilation
        # Use auto for automatic mixed precision on GPU
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
    
    # Calculate metrics using PyTorch (no sklearn needed)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    actuals_tensor = torch.tensor(actuals, dtype=torch.float32)
    
    from torchmetrics import MeanSquaredError, R2Score
    mse_metric = MeanSquaredError()
    r2_metric = R2Score()
    
    mse = mse_metric(predictions_tensor.squeeze(), actuals_tensor.squeeze()).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(predictions_tensor - actuals_tensor)).item()
    r2 = r2_metric(predictions_tensor.squeeze(), actuals_tensor.squeeze()).item()
    
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
