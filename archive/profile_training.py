"""
Complete profiling script to identify bottlenecks in TFT training.
Run this to see exactly where time is being spent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import cProfile
import pstats
import io
from contextlib import contextmanager

# Import your modules
from architecture_with_tft import TimeSeriesLSTM, AttentionLSTM, TemporalFusionTransformer
from dataset import make_dataloaders
from config_with_tft import TrainingConfig


# ============================================================================
# PROFILING UTILITIES
# ============================================================================

class TimingContext:
    """Context manager to time code blocks"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.4f}s")


class ProfileTracker:
    """Track timing for different parts of training"""
    def __init__(self):
        self.times = {
            'data_loading': [],
            'forward_pass': [],
            'loss_computation': [],
            'backward_pass': [],
            'optimizer_step': [],
            'validation': [],
            'logging': [],
        }
    
    def add(self, key, value):
        if key in self.times:
            self.times[key].append(value)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        for key, values in self.times.items():
            if values:
                avg = np.mean(values)
                total = np.sum(values)
                print(f"{key:20s} | Avg: {avg:.4f}s | Total: {total:.2f}s | Count: {len(values)}")
        print("="*60 + "\n")


# ============================================================================
# PROFILING TESTS
# ============================================================================

def profile_data_loading(config: TrainingConfig):
    """Profile how long data loading takes"""
    print("\n" + "="*60)
    print("PROFILING: DATA LOADING")
    print("="*60)
    
    train_loader, _, _ = make_dataloaders(pd.read_parquet('data/train.parquet'), config.window_size, config.batch_size)
    
    # Time loading all batches
    with TimingContext("Complete data loading (1 epoch)"):
        for batch_idx, (x, y) in enumerate(train_loader):
            pass  # Just iterate, don't process
    
    num_batches = len(train_loader)
    batch_size = config.batch_size
    print(f"Total batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Total samples: {num_batches * batch_size}")
    print(f"Avg time per batch: {(time.time() - time.time()) / num_batches:.4f}s")


def profile_forward_pass(config: TrainingConfig, device: torch.device, num_iterations: int = 100):
    """Profile model forward pass"""
    print("\n" + "="*60)
    print("PROFILING: FORWARD PASS")
    print("="*60)
    
    # Create model
    if config.model_type == 'lstm':
        model = TimeSeriesLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            fc_size=config.fc_size,
            num_layers=config.num_layers
        )
    elif config.model_type == 'tft':
        model = TemporalFusionTransformer(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout
        )
    
    model = model.to(device)
    model.eval()
    
    # Create sample input
    x_sample = torch.randn(config.batch_size, config.window_size, config.input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_sample)
    
    # Profile
    torch.cuda.synchronize() if device.type == 'cuda' else None
    with TimingContext(f"Forward pass ({num_iterations} iterations)"):
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    avg_time = time.time() / num_iterations
    print(f"Avg time per forward pass: {avg_time:.4f}s")
    print(f"Throughput: {config.batch_size / avg_time:.1f} samples/sec")


def profile_backward_pass(config: TrainingConfig, device: torch.device, num_iterations: int = 50):
    """Profile backward pass (forward + backward + optimizer step)"""
    print("\n" + "="*60)
    print("PROFILING: BACKWARD PASS")
    print("="*60)
    
    # Create model
    if config.model_type == 'lstm':
        model = TimeSeriesLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            fc_size=config.fc_size,
            num_layers=config.num_layers
        )
    elif config.model_type == 'tft':
        model = TemporalFusionTransformer(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout
        )
    
    model = model.to(device)
    model.train()
    
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create sample data
    x_sample = torch.randn(config.batch_size, config.window_size, config.input_size, device=device)
    y_sample = torch.randn(config.batch_size, config.input_size, device=device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(x_sample)
        loss = criterion(output, y_sample)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Profile complete backward pass
    with TimingContext(f"Forward + Backward + Optimizer ({num_iterations} iterations)"):
        for _ in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)
            output = model(x_sample)
            loss = criterion(output, y_sample)
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None


def profile_training_step(config: TrainingConfig, device: torch.device):
    """Profile individual components of a training step"""
    print("\n" + "="*60)
    print("PROFILING: INDIVIDUAL TRAINING COMPONENTS")
    print("="*60)
    
    # Create model
    if config.model_type == 'lstm':
        model = TimeSeriesLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            fc_size=config.fc_size,
            num_layers=config.num_layers
        )
    elif config.model_type == 'tft':
        model = TemporalFusionTransformer(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout
        )
    
    model = model.to(device)
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Get real data
    train_loader, _, _ = make_dataloaders(pd.read_parquet('data/train.parquet'), config.window_size, config.batch_size)
    
    tracker = ProfileTracker()
    num_batches = min(100, len(train_loader))  # Profile first 100 batches
    
    print(f"\nProfiling {num_batches} training batches...\n")
    
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # Data transfer
        with TimingContext("Data transfer to GPU"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        tracker.add('data_loading', time.time() - time.time())  # Placeholder
        
        # Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        output = model(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        forward_time = time.time() - start
        tracker.add('forward_pass', forward_time)
        
        # Loss computation
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        loss = criterion(output, y)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        loss_time = time.time() - start
        tracker.add('loss_computation', loss_time)
        
        # Backward pass
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        loss.backward()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        backward_time = time.time() - start
        tracker.add('backward_pass', backward_time)
        
        # Optimizer step
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        opt_time = time.time() - start
        tracker.add('optimizer_step', opt_time)
        
        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {batch_idx + 1}/{num_batches} batches")
    
    tracker.print_summary()


def profile_with_cprofile(config: TrainingConfig, device: torch.device, num_batches: int = 50):
    """Use Python's cProfile for detailed function profiling"""
    print("\n" + "="*60)
    print("PROFILING: DETAILED FUNCTION PROFILING (cProfile)")
    print("="*60)
    
    def train_subset():
        # Create model
        if config.model_type == 'lstm':
            model = TimeSeriesLSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                fc_size=config.fc_size,
                num_layers=config.num_layers
            )
        elif config.model_type == 'tft':
            model = TemporalFusionTransformer(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_lstm_layers=config.num_lstm_layers,
                dropout=config.dropout
            )
        
        model = model.to(device)
        model.train()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        train_loader, _, _ = make_dataloaders(pd.read_parquet('data/train.parquet'), config.window_size, config.batch_size)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    train_subset()
    profiler.disable()
    
    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions
    print(s.getvalue())


def main():
    """Run all profiling tests"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    config = TrainingConfig('tft')
    print(f"Model: {config.model_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Window size: {config.window_size}")
    print(f"Input size: {config.input_size}")
    
    # Run all profiling tests
    profile_data_loading(config)
    profile_forward_pass(config, device, num_iterations=100)
    profile_backward_pass(config, device, num_iterations=50)
    profile_training_step(config, device)
    profile_with_cprofile(config, device, num_batches=20)
    
    print("\n✅ Profiling complete!")


if __name__ == '__main__':
    main()