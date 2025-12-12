import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



# ============================================================================
# TFT COMPONENTS
# ============================================================================

class GatedResidualNetwork(nn.Module):
    """Optimized GRN - compute gate inputs only once"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Context processing if provided
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        # Gating mechanism - takes hidden state, not recomputing fc1
        self.gate_fc = nn.Linear(hidden_size, output_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None
    
    def forward(self, x, context=None):
        # Primary path - compute fc1 once
        a = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            a = a + self.context_fc(context)
        
        a = F.elu(a)
        a = self.fc2(a)
        a = self.dropout(a)
        
        # Gating - reuse hidden state, don't recompute fc1
        g = self.gate_fc(a)
        g = torch.sigmoid(g)
        
        # Gated output
        output = g * a
        
        # Skip connection
        if self.skip_layer is not None:
            output = output + self.skip_layer(x)
        else:
            output = output + x
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for TFT"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attn_weights


# ============================================================================
# PYTORCH LIGHTNING MODULE FOR TFT
# ============================================================================

class TemporalFusionTransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Temporal Fusion Transformer
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        num_lstm_layers: Number of LSTM layers
        dropout: Dropout probability
        output_size: Number of output features (defaults to input_size)
        learning_rate: Learning rate for optimizer
        lr_scheduler_factor: Factor for ReduceLROnPlateau
        lr_scheduler_patience: Patience for ReduceLROnPlateau
    """
    
    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_heads=2,
        num_lstm_layers=1,
        dropout=0.1,
        output_size=None,
        learning_rate=0.0005,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=5
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
       
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size if output_size else input_size
        self.learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        
        # Simple input projection instead of expensive VSN
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.self_attention = MultiHeadAttention(
            d_model=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            output_size=hidden_size * 2,
            dropout=dropout
        )
        
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            output_size=hidden_size * 2,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size * 2, self.output_size)

        # Minimal layer norm and dropout adapted for bidirectional output size
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Loss function
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Simple projection instead of expensive VSN
        x_proj = self.input_projection(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_proj)
        
        # Self-attention with residual
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        
        enriched = self.enrichment_grn(last_output)
        
        # Final output processing
        final_output = self.output_grn(enriched)
        predictions = self.output_layer(final_output)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # detach and move to cpu for sklearn
        y_np = y.detach().cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()

        # store for epoch-end metrics
        if not hasattr(self, "val_targets"):
            self.val_targets = []
            self.val_preds = []
        self.val_targets.append(y_np)
        self.val_preds.append(y_hat_np)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}
        
    def on_validation_epoch_end(self):
        # concatenate all batches
        if hasattr(self, "val_targets") and len(self.val_targets) > 0:
            y_true = np.concatenate(self.val_targets, axis=0)
            y_pred = np.concatenate(self.val_preds, axis=0)

            # flatten if necessary (for single-output regression)
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)

            r2 = r2_score(y_true_flat, y_pred_flat)

            # log and print R2
            self.log('val_r2', r2, prog_bar=True)
            print(f"\nEpoch {self.current_epoch} - Val R^2: {r2:.4f}")

            # clear for next epoch
            self.val_targets = []
            self.val_preds = []

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'predictions': y_hat, 'targets': y}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4  # regularization
        )
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# ============================================================================
# METRICS CALLBACK
# ============================================================================
    
class MetricsCallback(pl.Callback):
    """Custom callback to compute additional metrics during validation"""
    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = trainer.callback_metrics
        val_loss = outputs.get('val_loss', None)
        val_r2 = outputs.get('val_r2', None)
        current_lr = trainer.optimizers[0].param_groups[0]['lr']

        msg = "\n"
        if val_loss is not None:
            msg += f"Validation Loss: {val_loss.item():.6f} | "
        if val_r2 is not None:
            msg += f"Val R^2: {val_r2.item():.4f} | "
        msg += f"LR: {current_lr:.6f}"
        print(msg)
