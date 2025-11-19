from torch import nn
import torch
import torch.nn.functional as F

# ============================================================================
# ORIGINAL MODELS (TimeSeriesLSTM and AttentionLSTM)
# ============================================================================

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fc_size, num_layers, output_size=None):
        """
        Improved LSTM model for time series prediction with better regularization
        
        Args:
            input_size: Number of features in input
            hidden_size: Size of hidden state in LSTM
            fc_size: Size of fully connected layer
            num_layers: Number of LSTM layers
            output_size: Size of output (if None, uses input_size)
        """
        super(TimeSeriesLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size if output_size else input_size
        
        # LSTM layer with proper dropout configuration
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Add dropout for single-layer LSTM (applied to LSTM output)
        self.lstm_dropout = nn.Dropout(0.2) if num_layers == 1 else None
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(0.2)  # Dropout after ReLU
        self.fc2 = nn.Linear(fc_size, self.output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply dropout to LSTM output if single-layer
        if self.lstm_dropout is not None:
            lstm_out = self.lstm_dropout(lstm_out)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        
        return out


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, attention_dropout=0.3):
        """
        Attention mechanism with proper regularization
        
        Args:
            hidden_size: Size of LSTM hidden state
            attention_dropout: Dropout probability for attention weights
        """
        super(AttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dropout = attention_dropout
        
        # Learnable attention weights
        self.attn_weights = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Compute attention weights
        attn_weights = torch.tanh(self.attn_weights)
        
        # Apply attention weights to each timestep
        weighted = lstm_out * attn_weights
        
        # Apply dropout to prevent attention overfitting
        weighted = torch.nn.functional.dropout(
            weighted, 
            p=self.attention_dropout, 
            training=self.training
        )
        
        # Sum across time dimension to get context vector
        context = weighted.sum(dim=1)
        
        return context


class AttentionLSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        fc_size, 
        num_layers, 
        output_size=None,
        lstm_dropout=0.2,
        attention_dropout=0.3,
        fc_dropout=0.2
    ):
        """
        Simplified LSTM with attention mechanism and proper regularization
        
        Args:
            input_size: Number of features in input
            hidden_size: Size of hidden state in LSTM
            fc_size: Size of fully connected layer
            num_layers: Number of LSTM layers
            output_size: Size of output (if None, uses input_size)
            lstm_dropout: Dropout probability for LSTM output
            attention_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for FC layers
        """
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size if output_size else input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
        )
        
        # Dropout for single-layer LSTM
        self.lstm_dropout = nn.Dropout(lstm_dropout) if num_layers == 1 else None
        
        # Attention mechanism
        self.attention_block = AttentionBlock(hidden_size, attention_dropout)
        
        # Prediction head with LayerNorm and GELU
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.gelu = nn.GELU()
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc_size, self.output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply dropout to LSTM output if single-layer
        if self.lstm_dropout is not None:
            lstm_out = self.lstm_dropout(lstm_out)
        
        # Apply attention to get context vector
        context = self.attention_block(lstm_out)
        
        # Normalize and pass through prediction head
        context = self.layer_norm(context)
        out = self.fc1(context)
        out = self.gelu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_attention_regularization(self, l2_weight=0.001):
        """
        Compute L2 regularization loss for attention weights
        
        Args:
            l2_weight: Weight for L2 regularization term
            
        Returns:
            Regularization loss tensor
        """
        attn_reg = l2_weight * torch.sum(self.attention_block.attn_weights ** 2)
        return attn_reg


# ============================================================================
# TEMPORAL FUSION TRANSFORMER (TFT) COMPONENTS
# ============================================================================

class OldGatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from TFT paper"""
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
        
        # Gating mechanism
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
        # Primary path
        a = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            a = a + self.context_fc(context)
        
        a = F.elu(a)
        a = self.fc2(a)
        a = self.dropout(a)
        
        # Gating
        g = self.gate_fc(F.elu(self.fc1(x)))
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


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature selection"""
    def __init__(self, input_size, num_features, hidden_size, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Feature-wise GRNs
        self.grn_per_feature = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_features)
        ])
        
        # Softmax weights GRN
        self.grn_weights = GatedResidualNetwork(
            input_size * num_features,
            hidden_size,
            num_features,
            dropout
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features, input_size)
        batch_size, seq_len, num_features, _ = x.shape
        
        # Flatten features for weight computation
        x_flat = x.reshape(batch_size, seq_len, -1)
        
        # Compute feature weights
        weights = self.grn_weights(x_flat)
        weights = F.softmax(weights, dim=-1)
        
        # Apply feature-wise GRNs
        processed_features = []
        for i in range(num_features):
            feature_output = self.grn_per_feature[i](x[:, :, i, :])
            processed_features.append(feature_output)
        
        # Stack and weight features
        processed_features = torch.stack(processed_features, dim=2)
        weights = weights.unsqueeze(-1)
        
        # Weighted sum
        output = (processed_features * weights).sum(dim=2)
        
        return output, weights.squeeze(-1)


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


class OldTemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multivariate time series forecasting
    
    Simplified implementation suitable for your use case
    """
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_heads=4,
        num_lstm_layers=1,
        dropout=0.1,
        output_size=None
    ):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size if output_size else input_size
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Variable selection network
        # Treat each input feature as a "variable"
        self.vsn = VariableSelectionNetwork(
            input_size=1,
            num_features=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        # Position-wise feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # Output layers
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Reshape for VSN: (batch_size, seq_len, num_features, 1)
        x_vsn = x.unsqueeze(-1)
        
        # Variable selection
        x_selected, feature_weights = self.vsn(x_vsn)
        # x_selected shape: (batch_size, seq_len, hidden_size)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x_selected)
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Self-attention with residual
        attn_out, attn_weights = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        lstm_out = self.norm1(lstm_out + attn_out)
        
        # Post-attention GRN
        grn_out = self.post_attn_grn(lstm_out)
        grn_out = self.norm2(grn_out + lstm_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(grn_out)
        output = self.norm3(grn_out + ffn_out)
        
        # Use last timestep for prediction
        last_output = output[:, -1, :]
        
        # Final output processing
        final_output = self.output_grn(last_output)
        predictions = self.output_layer(final_output)
        
        return predictions
    
    def get_feature_importance(self, x):
        """
        Get feature importance weights for interpretability
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Feature importance weights
        """
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            x_vsn = x.unsqueeze(-1)
            _, feature_weights = self.vsn(x_vsn)
            # Average over batch and sequence
            avg_weights = feature_weights.mean(dim=[0, 1])
            return avg_weights

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
        g = self.gate_fc(a)  # FIXED: use 'a' instead of recomputing
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


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT without expensive VSN for faster training.
    Still captures temporal and feature interactions with minimal overhead.
    """
    def __init__(
        self,
        input_size,
        hidden_size=32,  # Reduced default
        num_heads=2,
        num_lstm_layers=1,
        dropout=0.1,
        output_size=None
    ):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size if output_size else input_size
        
        # Simple input projection instead of expensive VSN
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Single GRN instead of multiple
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        
        # Minimal layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
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
        
        # Final output processing
        final_output = self.output_grn(last_output)
        predictions = self.output_layer(final_output)
        
        return predictions
