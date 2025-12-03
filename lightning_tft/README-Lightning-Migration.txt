# PyTorch Lightning TFT Implementation

This folder contains a PyTorch Lightning conversion of the Temporal Fusion Transformer (TFT) model. The Lightning version provides cleaner code organization, automatic device handling, and better experiment tracking.

## Files Created

1. **tft_lightning_model.py** - Lightning module containing:
   - `GatedResidualNetwork` - TFT component
   - `MultiHeadAttention` - Attention mechanism
   - `TemporalFusionTransformerLightning` - Main Lightning module
   - `MetricsCallback` - Custom callback for metrics

2. **train_tft_lightning.py** - Training script with:
   - `train_tft_lightning()` - Main training function
   - `predict_with_model()` - Inference function
   - Automatic device selection (GPU/MPS/CPU)
   - TensorBoard logging

3. **config_lightning.py** - Configuration class:
   - `LightningTrainingConfig` - All hyperparameters

## Key Changes from Native PyTorch

### 1. Model Structure
**Before (Native PyTorch):**
```python
class TemporalFusionTransformer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return predictions
```

**After (PyTorch Lightning):**
```python
class TemporalFusionTransformerLightning(pl.LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.save_hyperparameters()  # Auto-save hyperparameters
        # Define layers
    
    def forward(self, x):
        # Forward pass (same as before)
        return predictions
    
    def training_step(self, batch, batch_idx):
        # Lightning handles device placement automatically
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Automatic validation loop
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        # Define optimizer and scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, ...)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
```

### 2. Training Loop
**Before (Native PyTorch):**
```python
# Manual training loop with 100+ lines
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)  # Manual device placement
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Manual validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # ... validation code
    
    # Manual early stopping
    # Manual checkpointing
    # Manual logging
```

**After (PyTorch Lightning):**
```python
# Clean 3-line training
trainer = pl.Trainer(max_epochs=100, callbacks=[...])
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
```

### 3. Device Handling
**Before:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x, y = x.to(device), y.to(device)  # In every batch
```

**After:**
```python
# Lightning handles automatically
trainer = pl.Trainer(accelerator='auto', devices=1)
# No manual .to(device) needed!
```

### 4. Checkpointing & Early Stopping
**Before:**
```python
# Custom EarlyStopping class (50+ lines)
# Manual checkpoint saving
if val_loss < best_loss:
    torch.save(model.state_dict(), 'best_model.pth')
```

**After:**
```python
# Built-in callbacks
checkpoint = ModelCheckpoint(monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=20)
trainer = pl.Trainer(callbacks=[checkpoint, early_stop])
```

### 5. Logging
**Before:**
```python
# Manual logging setup
logger = setup_logging(checkpoint_dir)
logger.info(f"Epoch {epoch}: loss={loss}")
```

**After:**
```python
# Built-in TensorBoard logging
self.log('train_loss', loss)  # In training_step
# View with: tensorboard --logdir checkpoints_lightning/tft_logs
```

## Usage

### Installation
```bash
pip install pytorch-lightning tensorboard
```

### Training
```python
from train_tft_lightning import train_tft_lightning
from config_lightning import LightningTrainingConfig

# Create config
config = LightningTrainingConfig()

# Customize if needed
config.num_epochs = 50
config.learning_rate = 0.001

# Train
model, trainer = train_tft_lightning(config)
```

Or run directly:
```bash
python train_tft_lightning.py
```

### Viewing Training Progress
```bash
tensorboard --logdir checkpoints_lightning/tft_logs
```
Then open http://localhost:6006 in your browser.

### Loading and Inference
```python
from train_tft_lightning import predict_with_model
from dataset import make_dataloaders

# Load data
train_loader, val_loader, test_loader = make_dataloaders(data, 100, 256)

# Make predictions
predictions, actuals = predict_with_model(
    'checkpoints_lightning/tft-epoch=10-val_loss=0.001234.ckpt',
    test_loader,
    device='cuda'
)
```

## Benefits of Lightning Version

### 1. Less Boilerplate
- **Before**: ~500 lines (train_with_tft.py)
- **After**: ~350 lines across 3 files (better organized)

### 2. Automatic Features
- ✅ Device placement (GPU/CPU/MPS)
- ✅ Mixed precision training (16-bit)
- ✅ Gradient clipping
- ✅ Progress bars
- ✅ Checkpointing
- ✅ Early stopping
- ✅ TensorBoard logging
- ✅ Learning rate monitoring

### 3. Better Organization
- Model code separated from training logic
- Easy to experiment with different configurations
- Cleaner callbacks instead of custom classes

### 4. Reproducibility
- `pl.seed_everything(42)` handles all random seeds
- `save_hyperparameters()` auto-saves model config
- Easy to load exact model from checkpoint

### 5. Multi-GPU Ready
```python
# To use multiple GPUs, just change:
trainer = pl.Trainer(accelerator='gpu', devices=4)
# That's it! Lightning handles distributed training.
```

## Configuration Options

Edit `config_lightning.py` to customize:

```python
class LightningTrainingConfig:
    # Model architecture
    hidden_size = 32          # Hidden dimension
    num_heads = 4             # Attention heads
    num_lstm_layers = 1       # LSTM layers
    dropout = 0.1             # Dropout rate
    
    # Training
    num_epochs = 100          # Max epochs
    learning_rate = 0.0005    # Initial LR
    batch_size = 256          # Batch size
    
    # Early stopping
    early_stopping_patience = 20
    early_stopping_min_delta = 1e-5
    
    # LR scheduling
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 8
```

## Comparison with Original

| Feature | Native PyTorch | PyTorch Lightning |
|---------|----------------|-------------------|
| Lines of code | ~500 | ~350 |
| Device handling | Manual | Automatic |
| Training loop | Manual | Automatic |
| Validation loop | Manual | Automatic |
| Early stopping | Custom class | Built-in |
| Checkpointing | Manual | Built-in |
| Logging | Custom | TensorBoard built-in |
| Mixed precision | Manual | Automatic |
| Multi-GPU | Complex | 1-line change |
| Reproducibility | Manual seed setting | `seed_everything()` |
| Model loading | `torch.load()` + setup | `load_from_checkpoint()` |

## Migration Guide

If you have existing trained models from the native PyTorch version:

1. **Convert weights** (if needed):
```python
# Load old model
old_state_dict = torch.load('old_model.pth')

# Create new Lightning model
new_model = TemporalFusionTransformerLightning(...)

# Load weights (Lightning adds 'model.' prefix sometimes)
new_model.load_state_dict(old_state_dict, strict=False)

# Save as Lightning checkpoint
trainer = pl.Trainer()
trainer.save_checkpoint("converted_model.ckpt")
```

2. **Use dataset.py unchanged** - DataLoader works the same way

3. **Update training scripts** - Use `train_tft_lightning.py` instead

## Notes

- The dataset.py file remains **unchanged** - PyTorch Lightning works with standard PyTorch DataLoaders
- The TFT architecture is **identical** - only the training wrapper changed
- Checkpoints are saved in a different format (`.ckpt` instead of `.pth`) but contain more information (optimizer state, epoch, etc.)
- Mixed precision is enabled automatically on CUDA devices for faster training

## Troubleshooting

**Q: Import error for pytorch_lightning**
```bash
pip install pytorch-lightning
```

**Q: Where are checkpoints saved?**
A: In `checkpoints_lightning/` directory by default

**Q: How to disable mixed precision?**
```python
trainer = pl.Trainer(..., precision='32')
```

**Q: How to continue training from checkpoint?**
```python
trainer.fit(model, train_loader, val_loader, ckpt_path='path/to/checkpoint.ckpt')
```
