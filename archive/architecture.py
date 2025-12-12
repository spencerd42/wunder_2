from torch import nn
import torch

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
        # lstm_out[:, -1, :] shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.fc_dropout(out)  # Apply dropout
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
        attn_weights = torch.tanh(self.attn_weights)  # (hidden_size,)
        
        # Apply attention weights to each timestep
        weighted = lstm_out * attn_weights  # Broadcasting
        
        # Apply dropout to prevent attention overfitting
        weighted = torch.nn.functional.dropout(
            weighted, 
            p=self.attention_dropout, 
            training=self.training
        )
        
        # Sum across time dimension to get context vector
        context = weighted.sum(dim=1)  # (batch_size, hidden_size)
        
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
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Apply dropout to LSTM output if single-layer
        if self.lstm_dropout is not None:
            lstm_out = self.lstm_dropout(lstm_out)
        
        # Apply attention to get context vector
        context = self.attention_block(lstm_out)
        # context shape: (batch_size, hidden_size)
        
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