from torch import nn

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fc_size, num_layers, output_size=None):
        """
        LSTM model for time series prediction
        
        Args:
            input_size: Number of features in input
            hidden_size: Size of hidden state in LSTM
            num_layers: Number of LSTM layers
            output_size: Size of output (if None, uses input_size)
        """
        super(TimeSeriesLSTM, self).__init__()
        
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
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, self.output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through fully connected layers
        fc1_out = self.relu(self.fc1(last_output))
        output = self.fc2(fc1_out)
        
        return output