import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ============================================================================
# TFT COMPONENTS
# ============================================================================

class GatedResidualNetwork(nn.Module):
    """Optimized GRN - compute gate inputs only once"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super(GatedResidualNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = 64
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
        a = self.fc1(x)
        
        # Context addition before activation
        if context is not None and self.context_size is not None:
            a = a + self.context_fc(context)
        
        a = F.elu(a)
        
        # ---- Gating now uses hidden state ----
        g = torch.sigmoid(self.gate_fc(a))
        
        # Now project and dropout
        a = self.fc2(a)
        a = self.dropout(a)
        
        output = g * a
        
        if self.skip_layer is not None:
            output = output + self.skip_layer(x)
        else:
            output = output + x
        
        output = self.layer_norm(output)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Fixed VSN that properly computes per-feature importance weights
    at each timestep, then combines weighted features into a condensed representation.
    
    Key improvements:
    1. Processes each feature through its own GRN to maintain feature-specific transformations
    2. Softmax over features (not time steps) ensures proper normalization
    3. Returns (batch, seq_len, hidden_size) preserving temporal dimension
    4. Weights are learned per timestep, not globally
    """
    
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Individual GRN for each feature - learns feature-specific transformations
        # This maintains the ability to learn which features matter
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1,  # Each GRN processes a single feature
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
            for _ in range(input_size)
        ])
        
        # Softmax over feature dimension to create importance weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Combine all feature representations into final hidden dimension
        self.feature_combiner = nn.Linear(input_size * hidden_size, hidden_size)
        
        # Optional: per-feature weight computation
        self.weight_grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            weighted_features: (batch_size, seq_len, hidden_size)
            weights: (batch_size, seq_len, input_size) - feature importance scores
        """
        batch_size, seq_len, input_size = x.shape
        
        # Process each feature through its own GRN
        # Shape: (batch, seq_len, hidden_size) for each feature
        feature_representations = []
        
        for i in range(self.input_size):
            feature = x[:, :, i:i+1]  # (batch, seq_len, 1)
            feature_repr = self.feature_grns[i](feature)  # (batch, seq_len, hidden_size)
            feature_representations.append(feature_repr)
        
        # Stack: (batch, seq_len, input_size, hidden_size)
        feature_representations = torch.stack(feature_representations, dim=2)
        
        # Compute importance weights per feature at each timestep
        weights = self.weight_grn(x)  # (batch, seq_len, input_size)
        weights = self.softmax(weights)  # Normalize across features
        
        # Apply weights to feature representations
        # Reshape weights for broadcasting: (batch, seq_len, input_size, 1)
        weights_expanded = weights.unsqueeze(-1)
        
        # Weighted sum: (batch, seq_len, hidden_size)
        weighted_features = (feature_representations * weights_expanded).sum(dim=2)
        
        return weighted_features, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for TFT"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
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
# Updated TFT Model with Fixed VSN
# ============================================================================

class TemporalFusionTransformerLightning(L.LightningModule):
    """TFT model with corrected VSN for sequence-level feature selection"""
    
    def __init__(
        self,
        input_size=32,
        hidden_size=64,
        num_heads=4,
        num_lstm_layers=2,
        dropout=0.1,
        output_size=32,
        learning_rate=0.0005,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=8,
    ):
        super(TemporalFusionTransformerLightning, self).__init__()
        
        self.save_hyperparameters()
        
        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        
        # Variable Selection Network (FIXED)
        self.vsn, self.vsn_weights = None, None
        self.variable_selection = VariableSelectionNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Encoder: LSTM to process historical context
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder: LSTM for future predictions
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Loss and metrics
        self.criterion = nn.MSELoss()
        
        # Storage for validation metrics
        self.val_targets = []
        self.val_preds = []
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            output: (batch_size, input_size)
        """
        # Variable Selection: learns which features matter at each timestep
        weighted_features, vsn_weights = self.variable_selection(x)
        
        # Store weights for potential analysis/logging
        self.vsn_weights = vsn_weights
        
        # Encoder: Process full sequence through LSTM
        encoder_output, (encoder_h, encoder_c) = self.encoder_lstm(weighted_features)
        
        # Attention: Focus on relevant timesteps
        attention_output, attention_weights = self.attention(
            query=encoder_output,
            key=encoder_output,
            value=encoder_output
        )
        
        # Decoder: Generate predictions (simplified - single step)
        # In practice, you'd expand this for multi-step forecasting
        decoder_input = attention_output[:, -1:, :]  # Last timestep
        decoder_output, _ = self.decoder_lstm(
            decoder_input,
            (encoder_h, encoder_c)
        )
        
        # Output projection
        output = self.fc_out(decoder_output[:, -1, :])  # (batch_size, output_size)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
        # Collect predictions and targets for R2 calculation
        self.val_preds.append(y_hat.detach())
        self.val_targets.append(y.detach())
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate R2 score
        if len(self.val_preds) > 0:
            y_pred = torch.cat(self.val_preds, dim=0).cpu().numpy()
            y_true = torch.cat(self.val_targets, dim=0).cpu().numpy()
            r2 = r2_score(y_true, y_pred)
            
            # Log and print R2
            self.log('val_r2', r2, prog_bar=True)
            print(f"\nEpoch {self.current_epoch} - Val R^2: {r2:.4f}")
        
        # Clear for next epoch
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

class MetricsCallback(L.Callback):
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