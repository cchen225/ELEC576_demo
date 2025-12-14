import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import math 
import sys

DATA_DIR = r'E:\Compiled\Curtis\School\Coolegg\Rice\PhD\Y2S1\Intro to Deep Learning\Final\data_processed'
MODEL_LOAD_PATH = './cnn_baseline_weights.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
DTYPE = torch.float32

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_height=128, input_width=44):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 
        )
        
        final_height = math.floor(input_height / (2**3)) 
        final_width = math.floor(input_width / (2**3))   
        flattened_size = 64 * final_height * final_width
        
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


def load_test_data(data_dir):
    """Loads saved numpy files for the test set."""
    try:
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        Y_test = np.load(os.path.join(data_dir, 'Y_test.npy'))
        
        X_test = np.expand_dims(X_test, axis=1) 
        
        X_test_tensor = torch.tensor(X_test, dtype=DTYPE)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        num_classes = Y_test.shape[1]
        input_height = X_test.shape[2]
        input_width = X_test.shape[3]

        return test_loader, num_classes, input_height, input_width

    except FileNotFoundError as e:
        print(f"Data files not found in the configured path: {data_dir}. {e}")
        return None, 0, 0, 0
    except Exception as e:
        print(f"Error occurred during data loading: {e}")
        return None, 0, 0, 0

def calculate_test_metrics(model, data_loader, instrument_names):
    """Calculates final metrics on the test set."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    criterion = nn.BCELoss() 
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating Test Set"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets) 
            total_loss += loss.item() * inputs.size(0)
            
            predictions = (outputs > 0.5).cpu().numpy()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions)
            
    avg_loss = total_loss / len(data_loader.dataset)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=instrument_names, zero_division=0))

    # Save final results to a markdown file
    report_path = os.path.join(DATA_DIR, 'final_test_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# Final Test Set Performance\n\n")
        f.write(f"**Overall Macro F1 Score:** {macro_f1:.4f}\n")
        f.write(f"**Test Loss (BCE):** {avg_loss:.4f}\n\n")
        f.write("## Per-Class Breakdown\n")
        f.write("```\n" + classification_report(all_targets, all_predictions, target_names=instrument_names, zero_division=0) + "\n```")
    print(f"Full report saved to {report_path}")

if __name__ == '__main__':
    # Retrieve instrument names from metadata file created by data_prep.py
    try:
        with open(os.path.join(DATA_DIR, 'metadata.txt'), 'r') as f:
            metadata = f.readlines()
        classes_line = [line for line in metadata if line.startswith('Classes:')][0]
        instrument_names = eval(classes_line.split(':', 1)[1].strip())
    except Exception:
        instrument_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']

    test_loader, num_classes, input_height, input_width = load_test_data(DATA_DIR)
    
    if test_loader is None:
        sys.exit()

    # Initialize model and load best weights
    model = SimpleCNN(num_classes=num_classes, input_height=input_height, input_width=input_width).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
        print(f"\nLoaded model weights from {MODEL_LOAD_PATH}.")
    except Exception as e:
        print(f"\nCould not load model weights.")
        print(e)
        sys.exit()

    calculate_test_metrics(model, test_loader, instrument_names)