import matplotlib.pyplot as plt
import numpy as np
import os
import sys

DATA_DIR = r'E:\Compiled\Curtis\School\Coolegg\Rice\PhD\Y2S1\Intro to Deep Learning\Final\data_processed'
OUTPUT_FILE = 'ssl_sl_comparison_plot_real.png'

def load_history(filename):
    """Loads a history dictionary from a .npy file."""
    path = os.path.join(DATA_DIR, filename)
    try:
        # allow_pickle=True is required to load the dictionary structure
        history = np.load(path, allow_pickle=True).item()
        return history
    except FileNotFoundError:
        print(f"Error: Could not find {filename} in {DATA_DIR}")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load CNN Baseline History (Supervised Learning - SL)
sl_history = load_history('training_history.npy')

# Load SSL Transformer History (Self-Supervised Learning - SSL)
ssl_history = load_history('ssl_finetune_history.npy')

if sl_history is None or ssl_history is None:
    print("Aborting plot generation due to missing data.")
    sys.exit()

# Extract Data
sl_train_loss = sl_history['train_loss']
sl_val_loss = sl_history['val_loss']
sl_f1_score = sl_history['val_f1']
sl_epochs = np.arange(1, len(sl_train_loss) + 1)

ssl_train_loss = ssl_history['train_loss']
ssl_val_loss = ssl_history['val_loss']
ssl_f1_score = ssl_history['val_f1']
ssl_epochs = np.arange(1, len(ssl_train_loss) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('SL vs. SSL Training Progression', fontsize=14, y=0.97)

ax1.set_title('Loss Convergence')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value (Binary Cross-Entropy)')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot CNN (SL)
ax1.plot(sl_epochs, sl_train_loss, label='SL Training Loss', color='blue', linestyle='-', marker='.')
ax1.plot(sl_epochs, sl_val_loss, label='SL Validation Loss', color='blue', linestyle='--', marker='.')

# Plot SSL
ax1.plot(ssl_epochs, ssl_train_loss, label='SSL Training Loss', color='red', linestyle='-', marker='.')
ax1.plot(ssl_epochs, ssl_val_loss, label='SSL Validation Loss', color='red', linestyle='--', marker='.')

ax1.legend(loc='upper right', fontsize=8)

ax2.set_title('F1 Score Progression (Validation)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro F1 Score')
ax2.grid(True, linestyle='--', alpha=0.6)

ax2.plot(sl_epochs, sl_f1_score, label='SL F1 Score', color='green', linewidth=2, marker='.')
ax2.plot(ssl_epochs, ssl_f1_score, label='SSL F1 Score', color='orange', linewidth=2, marker='.')
ax2.legend(loc='lower right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Comparison plot saved to {OUTPUT_FILE}")
plt.show()