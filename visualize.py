import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load walkable mask
im = Image.open("./train_data/walkable_mask.png")
walkable_mask = np.asarray(im, dtype=bool) == 0

# Load training walks
walks = pd.read_csv("./train_data/training_walks.csv")

# Create an RGB image for visualization
rssi_map = np.zeros((*walkable_mask.shape, 3), dtype=np.uint8)

# Function to normalize RSSI values to 0-255 range
def normalize_rssi(rssi_value):
    # Assuming RSSI values are typically between -100 and 0
    # Normalize to 0-255 range
    normalized = np.clip((rssi_value + 100) * 2.55, 0, 255)
    return np.uint8(normalized)

# Plot the first walk
plt.figure(figsize=(10, 10))
plt.imshow(walkable_mask)
