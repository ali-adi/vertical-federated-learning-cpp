#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Usage: plot_losses.py <loss_csv_path> <train_loss_image_path> <val_loss_image_path>")
    sys.exit(1)

loss_csv_path = sys.argv[1]
train_loss_image_path = sys.argv[2]
val_loss_image_path = sys.argv[3]

# Read the CSV file containing columns: epoch, train_loss, val_loss.
df = pd.read_csv(loss_csv_path)

# Use a built-in style ("ggplot") for a nicer look.
plt.style.use('ggplot')

# Smooth the curves using a rolling average (window size of 5 epochs, adjust if needed).
window_size = 5
df['train_loss_smooth'] = df['train_loss'].rolling(window=window_size, min_periods=1).mean()
df['val_loss_smooth'] = df['val_loss'].rolling(window=window_size, min_periods=1).mean()

# Plot Training Loss using the smoothed data.
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['train_loss_smooth'], label='Training Loss', color='royalblue', linewidth=3)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.title('Training Loss Over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(train_loss_image_path, format='jpeg', dpi=300)
plt.close()

# Plot Validation Loss using the smoothed data.
plt.figure(figsize=(12, 8))
plt.plot(df['epoch'], df['val_loss_smooth'], label='Validation Loss', color='crimson', linewidth=3)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.title('Validation Loss Over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(val_loss_image_path, format='jpeg', dpi=300)
plt.close()

print(f"Training loss plot saved to {train_loss_image_path}")
print(f"Validation loss plot saved to {val_loss_image_path}")
