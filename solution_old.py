import os
import sys
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint, ScorerStepByStep
from architecture import TimeSeriesLSTM
from config import TrainingConfig

class PredictionModel:
    """
    LSTM-based model for time series prediction.
    Maintains a sequence buffer and predicts the next value using the trained LSTM.
    Includes proper data scaling to match training conditions.
    """

    def __init__(self, model_path, input_size, hidden_size, fc_size, num_layers, window_size):
        self.current_seq_ix = None
        self.sequence_history = []
        self.window_size = window_size
        self.input_size = input_size
        
        # Load model
        self.device = torch.device('cpu')
        self.model = TimeSeriesLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            fc_size=fc_size,
            num_layers=num_layers,
            output_size=input_size
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")


    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset sequence history when we encounter a new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Scale and add current state to history
        state = data_point.state.copy()
        
        self.sequence_history.append(state)

        # Don't predict if not needed
        if not data_point.need_prediction:
            return None

        # Use the last window_size states
        window = np.array(self.sequence_history[-self.window_size:])
        
        # Convert to tensor
        x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        
        # Get prediction (in scaled space)
        with torch.no_grad():
            prediction = self.model(x)
        
        prediction = prediction.squeeze().cpu().numpy()
        
        return prediction


if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/data/train.parquet"
    
    config = TrainingConfig()

    # Create and test our model
    # Adjust input_size based on your dataset
    model = PredictionModel(
        model_path=f"{CURRENT_DIR}/checkpoints/best_model.pth",
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        fc_size=config.fc_size,
        num_layers=config.num_layers,
        window_size=config.window_size
    )

    # Load data into scorer
    scorer = ScorerStepByStep(test_file)

    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    # Evaluate our solution
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for features:")
    for i in range(len(scorer.features)):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)
