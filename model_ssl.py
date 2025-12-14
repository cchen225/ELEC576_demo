import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import math
import sys 

DATA_DIR = r'E:\Compiled\Curtis\School\Coolegg\Rice\PhD\Y2S1\Intro to Deep Learning\Final\data_processed'
PRETRAIN_MODEL_SAVE_PATH = './ssl_pretrain_weights.pt'
FINE_TUNE_MODEL_SAVE_PATH = './ssl_finetune_weights.pt'
DEVICE = torch.device("cpu") 

# Training Hyperparameters
SSL_LR = 1e-4
FINE_TUNE_LR = 1e-3
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
MASK_RATIO = 0.50 # 50% of tokens are masked during pre-training
PATCH_SIZE = (16, 16) # Patch size for the Transformer blocks

class PatchEmbedding(nn.Module):
    """Converts Log-Mel Spectrogram into sequences of patches (tokens)."""
    def __init__(self, input_height=128, input_width=44, patch_size=PATCH_SIZE, embed_dim=128):
        super().__init__()
        # Convolution to handle patch extraction (H, W) -> (H/P, W/P, embed_dim)
        self.patch_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.embed_dim = embed_dim
        
        H_out = math.floor((input_height - patch_size[0]) / patch_size[0]) + 1 
        W_out = math.floor((input_width - patch_size[1]) / patch_size[1]) + 1
        self.total_sequence_length = H_out * W_out # 8 * 2 = 16 patches
    
    def forward(self, x):
        # x shape: (B, 1, H, W)
        x = self.patch_conv(x)
        x = x.flatten(2) # (B, E, T)
        x = x.transpose(1, 2) # (B, T, E) -> (B, 16, 128)
        return x

class MaskedEncoder(nn.Module):
    """The Transformer Encoder used for SSL pre-training."""
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=256, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.transformer_encoder(x)

class SSLModel(nn.Module):
    """Two-Stage Model: Pre-training Encoder + Classification Head."""
    def __init__(self, num_classes, input_height=128, input_width=44):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(input_height=input_height, input_width=input_width)
        self.encoder = MaskedEncoder()
        
        self.reconstruction_head = nn.Linear(
            self.encoder.transformer_encoder.layers[0].self_attn.embed_dim, 
            self.encoder.transformer_encoder.layers[0].self_attn.embed_dim # Output size matches embedding size (128)
        )
        
        # Classification head for Stage 2 (Fine-tuning)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.transformer_encoder.layers[0].self_attn.embed_dim),
            nn.Linear(self.encoder.transformer_encoder.layers[0].self_attn.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward_encoder(self, x):
        """Passes input through the pre-training path (used in both stages)."""
        tokens = self.patch_embed(x)
        z = self.encoder(tokens)
        return z

def load_ssl_data(data_dir):
    """Loads and prepares the large UNLABELED dataset (X_ssl_full.npy) for pre-training."""
    try:
        X_ssl = np.load(os.path.join(data_dir, 'X_ssl_full.npy'))
        X_ssl = np.expand_dims(X_ssl, axis=1) # [N, 1, H, W]
        
        X_tensor = torch.tensor(X_ssl, dtype=torch.float32)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        input_height = X_ssl.shape[2]
        input_width = X_ssl.shape[3]
        return loader, input_height, input_width
    except Exception as e:
        print(f"Error loading SSL data (X_ssl_full.npy). Did you run data_prep.py with the SSL output enabled? Error: {e}")
        return None, 0, 0

def load_finetune_data(data_dir):
    """Loads the LABELED dataset for fine-tuning."""
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        Y_train = np.load(os.path.join(data_dir, 'Y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        Y_val = np.load(os.path.join(data_dir, 'Y_val.npy'))
        
        X_train = np.expand_dims(X_train, axis=1) 
        X_val = np.expand_dims(X_val, axis=1) 
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Retrieve instrument names from metadata file created by data_prep.py
        try:
            with open(os.path.join(data_dir, 'metadata.txt'), 'r') as f:
                metadata = f.readlines()
            classes_line = [line for line in metadata if line.startswith('Classes:')][0]
            instrument_names = eval(classes_line.split(':', 1)[1].strip())
        except Exception:
            instrument_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
            
        num_classes = Y_train.shape[1]
        return train_loader, val_loader, num_classes, instrument_names
    except Exception as e:
        print(f"Error loading Fine-tune data: {e}")
        return None, None, 0, None

# Stage 1: Pre-Training

def run_pretrain(model, ssl_loader, input_height, input_width):
    """
    Trains the encoder to perform Masked Autoencoding on unlabeled data.
    """
    print("\nStarting SSL Pre-training (Masked Autoencoding)")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=SSL_LR)
    criterion = nn.MSELoss()

    # Fetch the correct sequence length from the model's patch embedding layer
    token_len = model.patch_embed.total_sequence_length
    
    # Token mask generation utility
    def generate_mask(B):
        # Generate random indices to mask (hide)
        mask = torch.rand(B, token_len)
        mask = (mask < MASK_RATIO).bool().to(DEVICE)
        return mask

    for epoch in range(PRETRAIN_EPOCHS):
        total_loss = 0
        for data in tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}/{PRETRAIN_EPOCHS}"):
            inputs = data[0].to(DEVICE)
            
            # Create tokens and masks
            tokens = model.patch_embed(inputs) # (B, T, E) where T = 16
            mask = generate_mask(tokens.shape[0]) # (B, 16)
            
            # Apply mask (simple zeroing for quick implementation)
            masked_tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0) 
            
            # Encode and Reconstruct
            encoded_tokens = model.encoder(masked_tokens)
            reconstructed_patches = model.reconstruction_head(encoded_tokens) # (B, T, E=128)

            # Calculate Loss only on the masked tokens
            target_patches = tokens
            
            # Reshape tokens/patches to (B * T, E)
            target_patches_flat = target_patches.reshape(-1, target_patches.shape[-1])
            reconstructed_patches_flat = reconstructed_patches.reshape(-1, reconstructed_patches.shape[-1])
            mask_flat = mask.flatten()
            
            # Loss only on masked elements
            loss = criterion(reconstructed_patches_flat[mask_flat], target_patches_flat[mask_flat])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"SSL Epoch {epoch+1} Loss: {total_loss / len(ssl_loader):.4f}")
    
    torch.save(model.state_dict(), PRETRAIN_MODEL_SAVE_PATH)
    print(f"Pre-trained encoder weights saved to {PRETRAIN_MODEL_SAVE_PATH}")
    return model

# Stage 2: Fine-Tuning

def run_finetune(model, train_loader, val_loader, num_classes, instrument_names):
    """
    Attaches the classification head and fine-tunes the entire network.
    Saves validation history and best F1 score for visualization.
    """
    print("\nStarting Supervised Fine-tuning")
    
    # History tracking for visualization
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    # Freeze the encoder weights initially to speed up early training
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Train only the classifier head
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    criterion = nn.BCELoss()
    
    best_f1 = -1
    
    def calculate_metrics_finetune(model, loader):
        model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # Use encoder output for classification
                encoded_tokens = model.forward_encoder(inputs)
                pooled_token = torch.mean(encoded_tokens, dim=1) 
                
                outputs = model.classifier(pooled_token)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                predictions = (outputs > 0.5).cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions)
                
        avg_loss = total_loss / len(loader.dataset)
        macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        
        # Get final per-class breakdown for comparison with CNN baseline
        f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
        
        return avg_loss, macro_f1, f1_per_class

    
    for epoch in range(FINE_TUNE_EPOCHS):
        model.train()
        train_loss = 0
        
        # Unfreeze encoder and reduce LR after initial classifier training
        if epoch == 5: 
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR / 5.0) # Lower LR for stability
            print("Encoder Unfrozen. Fine-tuning entire network.")


        for inputs, targets in tqdm(train_loader, desc=f"Fine-Tune Epoch {epoch+1}/{FINE_TUNE_EPOCHS}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass through the pre-trained encoder
            encoded_tokens = model.forward_encoder(inputs)
            pooled_token = torch.mean(encoded_tokens, dim=1)
            
            outputs = model.classifier(pooled_token)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss, val_f1, f1_per_class = calculate_metrics_finetune(model, val_loader)
        
        # Update history arrays
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Fine-Tune Epoch {epoch+1} Loss: {train_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), FINE_TUNE_MODEL_SAVE_PATH)
            print(f"  >>> Best model saved (F1: {best_f1:.4f})")
            
            # Save the per-class F1 for the best epoch (Saves data for the final report comparison)
            best_ssl_f1_data = {
                'classes': instrument_names,
                'f1_scores': f1_per_class.tolist(),
                'macro_f1': best_f1
            }
            np.save(os.path.join(DATA_DIR, 'ssl_best_f1_per_class.npy'), best_ssl_f1_data)

    # Save the full fine-tuning history at the end
    np.save(os.path.join(DATA_DIR, 'ssl_finetune_history.npy'), history)
    
    # Load the best model weights for final evaluation
    model.load_state_dict(torch.load(FINE_TUNE_MODEL_SAVE_PATH))
    return model

# --- EXECUTION ---

if __name__ == '__main__':
    
    # Load Fine-tune data first to get input dimensions, classes, and names
    train_loader_ft, val_loader_ft, num_classes, instrument_names = load_finetune_data(DATA_DIR)
    
    if train_loader_ft is None:
        sys.exit()

    # Determine input shape from the first batch of data
    first_batch_ft = next(iter(train_loader_ft))
    input_height, input_width = first_batch_ft[0].shape[2], first_batch_ft[0].shape[3]
    
    # Load SSL data (using separate loader as it has no labels)
    ssl_loader, _, _ = load_ssl_data(DATA_DIR)
    
    # Initialize the SSL/Fine-tune model
    ssl_model = SSLModel(num_classes=num_classes, input_height=input_height, input_width=input_width).to(DEVICE)
    
    print("\nSSL MODEL SUMMARY (Transformer)")
    print(ssl_model)
    
    # Stage 1: Pre-training (Unsupervised)
    if ssl_loader is not None:
        ssl_model = run_pretrain(ssl_model, ssl_loader, input_height, input_width)
    
    # Stage 2: Fine-tuning (Supervised)
    final_ssl_model = run_finetune(ssl_model, train_loader_ft, val_loader_ft, num_classes, instrument_names)
    
    print("\nSSL Pipeline Complete")