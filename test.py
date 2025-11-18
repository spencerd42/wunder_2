import pandas as pd

from dataset import split_data

data = pd.read_parquet('data/train.parquet')

split_data(data)