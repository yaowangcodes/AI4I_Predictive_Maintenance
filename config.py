
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent

# Random seed
SEED = 42

# File paths
DATA = ROOT / "ai4i2020.csv"
OUT = ROOT / "artifacts"

OUT.mkdir(exist_ok=True)

# Target variable
TARGET = "Machine failure"

# Categorical features
CAT = ["Type"]

# Numerical features
NUM = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# All features
FEATURES = CAT + NUM

# MLflow configuration
TRACKING_URI = "http://127.0.0.1:5001"

EXPERIMENT_NAME = "AI4I_predictive_maintenance_V2"

MODEL_NAME = "AI4I_Predictive_Maintenance"
