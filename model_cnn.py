import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import math

DATA_DIR = r'E:\Compiled\Curtis\School\Coolegg\Rice\PhD\Y2S1\Intro to Deep Learning\Final\data_processed'
MODEL_SAVE_PATH = './cnn_baseline_weights.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 8
DTYPE = torch.float32

class SimpleCNN(nn.Module):
    """
    A simple 2D Convolutional Neural Network designed for Log-Mel Spectrogram input.
    The architecture mimics simple VGG-style blocks, common for audio classification tasks.
    """
    def __init__(self, num_classes, input_height=128, input_width=44):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Layers (Feature Extractor)
        self.conv_layers = nn.Sequential(
            # Input: [1, 128, 44]
            # Block 1
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # Output: [16, 64, 22]

            # Block 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # Output: [32, 32, 11]
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # Output: [64, 16, 5] 
        )
        
        # Calculate the size of the flattened feature map
        final_height = math.floor(input_height / (2**3)) 
        final_width = math.floor(input_width / (2**3))   
        
        flattened_size = 64 * final_height * final_width
        
        # Classifier Head
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return x

def load_data(data_dir):
    """Loads saved numpy files and converts them to PyTorch Tensors."""
    print("Loading data...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        Y_train = np.load(os.path.join(data_dir, 'Y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        Y_val = np.load(os.path.join(data_dir, 'Y_val.npy'))
        
        X_train = np.expand_dims(X_train, axis=1) 
        X_val = np.expand_dims(X_val, axis=1) 
        
        # Convert to PyTorch Tensors using float32
        X_train_tensor = torch.tensor(X_train, dtype=DTYPE)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=DTYPE)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        num_classes = Y_train.shape[1]
        input_height = X_train.shape[2]
        input_width = X_train.shape[3]

        return train_loader, val_loader, num_classes, input_height, input_width

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Data files not found in the configured path: {data_dir}")
        return None, None, 0, 0, 0
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None, 0, 0, 0

def calculate_metrics(model, data_loader, num_classes, instrument_names):
    """Calculates validation loss and Macro F1-Score"""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    criterion = nn.BCELoss() 
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets) 
            total_loss += loss.item() * inputs.size(0)
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            predictions = (outputs > 0.5).cpu().numpy()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions)
            
    avg_loss = total_loss / len(data_loader.dataset)
    
    # Calculate Macro F1-Score (unweighted mean of F1 per class)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    # Calculate Per-Class F1-Scores
    f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    report = classification_report(all_targets, all_predictions, target_names=instrument_names, zero_division=0, output_dict=True)

    return avg_loss, macro_f1, f1_per_class, report


def train_model(model, train_loader, val_loader, num_classes, instrument_names):
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = -1
    
    print(f"\nStarting Training on {DEVICE}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Training)"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation and Metrics
        val_loss, val_macro_f1, val_f1_per_class, val_report = calculate_metrics(
            model, val_loader, num_classes, instrument_names
        )
        
        # Update history arrays
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_macro_f1)
        
        print(f"Epoch {epoch+1} Complete: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")
        
        # Save the best model based on validation Macro F1-Score
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Model saved to {MODEL_SAVE_PATH} (F1: {best_f1:.4f})")
            
            # Save the per-class F1 for the best epoch
            per_class_data = {
                'classes': instrument_names,
                'f1_scores': val_f1_per_class.tolist(),
                'macro_f1': best_f1
            }
            np.save(os.path.join(DATA_DIR, 'best_f1_per_class.npy'), per_class_data)
            print(f"Per-class F1 data saved.")
    
    # Save the full training history at the end (Saves data for Figure 2)
    np.save(os.path.join(DATA_DIR, 'training_history.npy'), history)

    print("\nTraining Finished")
    print(f"Best Validation Macro F1: {best_f1:.4f}")
    return best_f1

if __name__ == '__main__':
    
    # Retrieve instrument names from metadata file created by data_prep.py
    try:
        with open(os.path.join(DATA_DIR, 'metadata.txt'), 'r') as f:
            metadata = f.readlines()
        
        classes_line = [line for line in metadata if line.startswith('Classes:')][0]
        instrument_names = eval(classes_line.split(':', 1)[1].strip())
    except Exception:
        instrument_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
        print("Warning: Could not read instrument names from metadata. Using generic names.")


    train_loader, val_loader, num_classes, input_height, input_width = load_data(DATA_DIR)
    
    if train_loader is None:
        exit()
        
    model = SimpleCNN(
        num_classes=num_classes, 
        input_height=input_height, 
        input_width=input_width
    ).to(DEVICE)
    
    print("\nModel Summary")
    print(model)
    print(f"Input Spectrogram Shape: (1, {input_height}, {input_width})")
    print(f"Output Classes: {instrument_names}")
    
    final_f1 = train_model(model, train_loader, val_loader, num_classes, instrument_names)

    print("\nTraining complete.")