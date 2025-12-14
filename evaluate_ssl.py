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
MODEL_LOAD_PATH = './ssl_finetune_weights.pt'
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
DTYPE = torch.float32 

PATCH_SIZE = (16, 16) 

class PatchEmbedding(nn.Module):
    def __init__(self, input_height=128, input_width=44, patch_size=PATCH_SIZE, embed_dim=128):
        super().__init__()
        self.patch_conv = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        H_out = math.floor((input_height - patch_size[0]) / patch_size[0]) + 1 
        W_out = math.floor((input_width - patch_size[1]) / patch_size[1]) + 1
        self.total_sequence_length = H_out * W_out
        self.embed_dim = embed_dim
    
    def forward(self, x):
        x = self.patch_conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MaskedEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        return self.transformer_encoder(x)

class SSLModel(nn.Module):
    def __init__(self, num_classes, input_height=128, input_width=44):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_height=input_height, input_width=input_width)
        self.encoder = MaskedEncoder()
        
        # Reconstruction head is irrelevant for evaluation, but must be defined to load state_dict
        patch_area_x_channels = self.patch_embed.patch_conv.out_channels * PATCH_SIZE[0] * PATCH_SIZE[1]
        self.reconstruction_head = nn.Linear(128, 128) 
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward_encoder(self, x):
        tokens = self.patch_embed(x)
        z = self.encoder(tokens)
        return z
    
    def forward(self, x):
        encoded_tokens = self.forward_encoder(x)
        pooled_token = torch.mean(encoded_tokens, dim=1) # Global average pooling
        output = self.classifier(pooled_token)
        return output

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
        print(f"Data not found in the path: {data_dir}. {e}")
        return None, 0, 0, 0
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, 0, 0, 0

def calculate_test_metrics(model, data_loader, instrument_names):
    """Calculates final metrics on the test set."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    criterion = nn.BCELoss() 
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating SSL Test Set"):
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
    print(classification_report(all_targets, all_predictions, target_names=instrument_names, zero_division=0))

    # Save final results to a file for comparison plotting
    ssl_test_data = {
        'classes': instrument_names,
        'f1_scores': f1_score(all_targets, all_predictions, average=None, zero_division=0).tolist(),
        'macro_f1': macro_f1
    }
    np.save(os.path.join(DATA_DIR, 'ssl_final_test_results.npy'), ssl_test_data)
    print(f"SSL Test results saved to ssl_final_test_results.npy")

if __name__ == '__main__':
    # Load instrument names
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
    model = SSLModel(num_classes=num_classes, input_height=input_height, input_width=input_width).to(DEVICE)
    try:
        # Load state dict (model weights)
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
        print(f"\nLoaded SSL Transformer weights from {MODEL_LOAD_PATH}.")
    except Exception as e:
        print(f"\nCould not load SSL model weights.")
        print(e)
        sys.exit()

    calculate_test_metrics(model, test_loader, instrument_names)