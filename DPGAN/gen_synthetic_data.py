import sys

if len(sys.argv) < 3:
    print("gen_synthetic_data.py ./datasets/{dataset}.csv {epsilon = (0.1, 1, 10)}")
    sys.exit(1)

DATASET = sys.argv[1]
EPSILON = sys.argv[2]
match DATASET:
    case "./datasets/OULADStudentInfo.csv":
        match EPSILON:
            case '0.1':
                BATCH_SIZE = 256
                EPOCHS = 8
                NOISE_MULTIPLIER = 1.5
            case '1':
                BATCH_SIZE = 256
                EPOCHS = 15
                NOISE_MULTIPLIER = 0.8
            case '10':
                BATCH_SIZE = 512
                EPOCHS = 25
                NOISE_MULTIPLIER = 0.3
            case _:
                print("gen_synthetic_data.py ./datasets/{dataset}.csv {epsilon = (0.1, 1, 10)}")
                sys.exit(1)
    case "./datasets/StudentsPerformanceExams.csv":
        match EPSILON:
            case '0.1':
                BATCH_SIZE = 512
                EPOCHS = 2
                NOISE_MULTIPLIER = 5
            case '1':
                BATCH_SIZE = 512
                EPOCHS = 8
                NOISE_MULTIPLIER = 1.5
            case '10':
                BATCH_SIZE = 512
                EPOCHS = 15
                NOISE_MULTIPLIER = 0.4
            case _:
                print("gen_synthetic_data.py ./datasets/{dataset}.csv {epsilon = (0.1, 1, 10)}")
                sys.exit(1)
    case "./datasets/USPHDStudentData.csv":
        match EPSILON:
            case '0.1':
                BATCH_SIZE = 256
                EPOCHS = 3
                NOISE_MULTIPLIER = 4
            case '1':
                BATCH_SIZE = 256
                EPOCHS = 10
                NOISE_MULTIPLIER = 1.5
            case '10':
                BATCH_SIZE = 256
                EPOCHS = 25
                NOISE_MULTIPLIER = 0.5
            case _:
                print("gen_synthetic_data.py ./datasets/{dataset}.csv {epsilon = (0.1, 1, 10)}")
                sys.exit(1)
    case _:
        print("gen_synthetic_data.py ./datasets/{dataset}.csv {epsilon = (0.1, 1, 10)}")
        sys.exit(1)
        



import logging
import os
        
import pandas as pd
import torch

from dpwgan import CategoricalDataset
from dpwgan.utils import create_categorical_gan

NOISE_DIM = 20
HIDDEN_DIM = 20


torch.manual_seed(123)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Preparing data set ({DATASET})...")
try:
    df = pd.read_csv(DATASET, dtype=str)
except FileNotFoundError:
    print(f"ERROR: Could not locate data set [{DATASET}]")
    sys.exit(1)

df = df.fillna('N/A')
dataset = CategoricalDataset(df)
data = dataset.to_onehot_flat()

gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)

logger.info(f"Training GAN (e = {sys.argv[1]})...")
gan.train(
    data=data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sigma=NOISE_MULTIPLIER
)

logger.info("Generating synthetic data...")
flat_synthetic_data = gan.generate(len(df))
synthetic_data = dataset.from_onehot_flat(flat_synthetic_data)

filename = "./synthetic_output.csv"
with open(filename, 'w') as f:
    synthetic_data.to_csv(f, index=False)

logger.info(f"Synthetic data saved to {filename}")
