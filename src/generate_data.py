import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

data = {
    "hours_slept": np.random.uniform(4, 9, 200),
    "sleep_quality": np.random.uniform(3, 10, 200),
    "exercise_hours": np.random.uniform(0, 2.5, 200),
    "caffeine": np.random.randint(0, 5, 200),
    "stress_level": np.random.uniform(1, 10, 200),
    "screen_time": np.random.uniform(2, 10, 200)
}

df = pd.DataFrame(data)

# Create energy score
df["energy_score"] = (
    df["hours_slept"] * 7
    + df["sleep_quality"] * 3
    + df["exercise_hours"] * 2
    - df["caffeine"] * 1
    - df["stress_level"] * 3
    - df["screen_time"] * 1.5
)

# Add a little randomness
df["energy_score"] += np.random.normal(0, 3, 200)

# Keep score between 0 and 100
df["energy_score"] = df["energy_score"].clip(0, 100)

# Round values
df = df.round(2)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sleep_energy_data.csv"

df.to_csv(DATA_PATH, index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print(f"Dataset saved to: {DATA_PATH}")