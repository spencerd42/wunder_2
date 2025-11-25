import os
import sys
import numpy as np
import torch
from tft_lightning_model import TemporalFusionTransformerLightning
from config_lightning import LightningTrainingConfig

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint, ScorerStepByStep

# OPTIMIZATION: Can speed up CPU inference
torch.set_flush_denormal(True) 

class PredictionModel:
    """
    OPTIMIZED LSTM-based model for time series prediction.
    - Uses a pre-allocated NumPy array as a ring buffer for high-speed windowing.
    - Uses torch.from_numpy for zero-copy tensor creation.
    - Assumes data is already scaled (or doesn't need scaling).
    """

    def __init__(self):
        config = LightningTrainingConfig()
        
        self.current_seq_ix = None
        self.window_size = config.window_size
        self.input_size = config.input_size
        
        # --- OPTIMIZATION: Pre-allocate buffer ---
        # This is VASTLY faster than appending to a Python list
        self.history_buffer = np.zeros((self.window_size, self.input_size), dtype=np.float32)
        self.history_count = 0  # Tracks items in buffer for new sequences
        
        # Load model
        self.device = torch.device('cpu')

        self.model_path = 'checkpoints_lightning/tft-epoch=16-val_loss=0.603450.ckpt'
        
        # Load checkpoint
        if os.path.exists(self.model_path):
            self.model = TemporalFusionTransformerLightning.load_from_checkpoint(self.model_path)
            self.model.eval()
            self.model.to('cpu')
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")


    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset sequence history when we encounter a new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            # Reset buffer counter
            self.history_count = 0

        # Get current state
        state = data_point.state.copy()

        # --- OPTIMIZATION: Add to pre-allocated buffer ---
        if self.history_count < self.window_size:
            # Fill the buffer until it's full
            self.history_buffer[self.history_count] = state
        else:
            # Once full, "roll" the buffer by shifting all rows up
            # and adding the new state at the end.
            self.history_buffer[:-1] = self.history_buffer[1:]
            self.history_buffer[-1] = state
            
        self.history_count += 1

        # Don't predict if not needed
        if not data_point.need_prediction:
            return None

        # --- LOGIC: Wait for a full buffer (matches training) ---
        # Don't predict if buffer isn't full (matches training logic)
        # The first valid prediction is when history_count == 100
        if self.history_count < self.window_size:
            return None

        # --- OPTIMIZATION: Use torch.from_numpy (zero-copy) ---
        # .unsqueeze(0) adds the batch dimension.
        x = torch.from_numpy(self.history_buffer).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(x)
        
        prediction = prediction.squeeze().cpu().numpy()
        
        return prediction


if __name__ == "__main__":
    # Check existence of test file
    test_file = f"{CURRENT_DIR}/data/train.parquet"
    
    config = LightningTrainingConfig()

    # Create and test our model
    # Adjust input_size based on your dataset
    model = PredictionModel()

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