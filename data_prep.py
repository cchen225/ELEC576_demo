import os
import glob
import random
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import soundfile as sf
import warnings
import yaml
from multiprocessing import Pool, cpu_count 
from itertools import repeat 

# Suppress librosa warnings (optional, but helpful for clean output)
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
# Using the path structure you provided
SLAKH_ROOT = 'E:/Compiled/Curtis/School/Coolegg/Rice/PhD/Y2S1/Intro to Deep Learning/Final/slakh2100_flac_redux' 
SUBDIRS = ['train', 'validation', 'test']

# Target instrument classes for multi-label classification (MUST BE CONSISTENT)
TARGET_CLASSES = ['Bass', 'Piano', 'Guitar', 'Drums', 'Organ', 'Strings'] 
INSTRUMENT_TO_INDEX = {inst: i for i, inst in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

# Audio processing parameters
SR = 22050           # Sample Rate
N_MELS = 128         # Number of Mel bands
N_FFT = 2048         
HOP_LENGTH = 512     
SAMPLE_DURATION = 1.0  
SAMPLES_PER_TRACK = 300 # Aggressive sampling
MAX_STEMS_PER_MIX = 3  

# Output parameters
# --- CRITICAL FIX: Changing output location to C drive (assuming more space) ---
OUTPUT_DIR = 'C:/temp/data_processed' 
TRAIN_RATIO = 0.8    

# --- FEATURE EXTRACTION FUNCTION ---

def audio_to_log_mel_spectrogram(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Converts a raw audio array into a Log-Mel Spectrogram (2D feature).
    Includes stability checks for silent audio.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
    
    min_val = log_mel_spectrogram.min()
    max_val = log_mel_spectrogram.max()
    data_range = max_val - min_val
    
    if data_range == 0:
        return np.zeros_like(log_mel_spectrogram)
    
    epsilon = 1e-6
    log_mel_spectrogram = (log_mel_spectrogram - min_val) / (data_range + epsilon)
    
    return log_mel_spectrogram

# --- SLAKH METADATA PARSING ---

def get_stem_map(track_dir):
    """ Reads metadata.yaml to map stem ID (e.g., 'S00') to instrument class ('Bass'). """
    metadata_path = os.path.join(track_dir, 'metadata.yaml')
    stem_map = {}
    
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)

    if 'stems' in metadata:
        for stem_id, stem_info in metadata['stems'].items():
            inst_class = stem_info.get('inst_class')
            audio_rendered = stem_info.get('audio_rendered', False)
            
            if inst_class in TARGET_CLASSES and audio_rendered:
                stem_map[stem_id] = inst_class
    return stem_map

# --- CORE PROCESSING FUNCTION FOR MULTIPROCESSING ---

def process_single_track(track_dir):
    """
    Generates SAMPLES_PER_TRACK polyphonic mixes from a single track directory.
    Returns a list of (spectrogram, label) tuples.
    """
    track_features = []
    
    stem_to_class_map = get_stem_map(track_dir)
    stems_dir = os.path.join(track_dir, 'stems') 

    available_stems = []
    for stem_id, inst_class in stem_to_class_map.items():
        file_path = os.path.join(stems_dir, f'{stem_id}.flac') 
        if os.path.exists(file_path):
            available_stems.append((file_path, inst_class))
    
    if len(available_stems) < 2:
        return [] 

    # Generate multiple polyphonic mixes
    for _ in range(SAMPLES_PER_TRACK):
        num_stems_to_mix = random.randint(2, min(MAX_STEMS_PER_MIX, len(available_stems)))
        mix_selection = random.sample(available_stems, num_stems_to_mix)
        
        mixed_audio = None
        current_label = np.zeros(NUM_CLASSES, dtype=np.float32)

        try:
            shortest_duration = float('inf')
            for path, _ in mix_selection:
                info = sf.info(path)
                shortest_duration = min(shortest_duration, info.duration)

            max_start_time = shortest_duration - SAMPLE_DURATION
            if max_start_time <= 0:
                continue 

            start_time = random.uniform(0, max_start_time)

            for path, inst_class in mix_selection:
                stem_audio, _ = librosa.load(
                    path, sr=SR, offset=start_time, duration=SAMPLE_DURATION
                )
                
                if mixed_audio is None:
                    mixed_audio = stem_audio
                else:
                    min_len = min(len(mixed_audio), len(stem_audio))
                    mixed_audio = mixed_audio[:min_len] + stem_audio[:min_len]
                
                current_label[INSTRUMENT_TO_INDEX[inst_class]] = 1.0
            
            mixed_audio = librosa.util.normalize(mixed_audio)
            spectrogram = audio_to_log_mel_spectrogram(mixed_audio)
            
            track_features.append((spectrogram, current_label))

        except Exception:
            continue
            
    return track_features

# --- MAIN DATA PROCESSING FUNCTION USING POOL ---

def process_slakh_subset_parallel(root_dir, subset_name):
    """
    Processes a single Slakh subset using multiprocessing.
    """
    subset_path = os.path.join(root_dir, subset_name)
    
    track_dirs = []
    if os.path.exists(subset_path):
        track_dirs = [os.path.join(subset_path, d) for d in os.listdir(subset_path) 
                      if os.path.isdir(os.path.join(subset_path, d)) and d.startswith('Track')]
    
    if not track_dirs:
        return np.array([]), np.array([])
        
    print(f"--- Processing {subset_name} ({len(track_dirs)} tracks) using {cpu_count()} cores ---")

    all_results = []
    
    # Use multiprocessing Pool to distribute the work
    # Note: Using tqdm with multiprocessing requires using Pool.imap and explicit list conversion
    with Pool(processes=cpu_count()) as pool:
        # pool.imap is used here for the progress bar support
        for result in tqdm(pool.imap(process_single_track, track_dirs), total=len(track_dirs), desc="Generating mixes"):
            all_results.extend(result)

    # Separate features and labels
    if all_results:
        subset_features = np.array([res[0] for res in all_results], dtype=np.float32)
        subset_labels = np.array([res[1] for res in all_results], dtype=np.float32)
    else:
        return np.array([]), np.array([])
    
    print(f"\nSuccessfully generated {subset_features.shape[0]} samples from {subset_name}.")
    return subset_features, subset_labels

# --- SPLITTING AND SAVING ---

def save_data(X, Y, output_dir=OUTPUT_DIR, subset_name='train'):
    """
    Saves the processed NumPy arrays with descriptive names.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save X and Y arrays
    # Removed allow_pickle=True since it's only needed for non-standard numpy objects
    np.save(os.path.join(output_dir, f'X_{subset_name}.npy'), X)
    np.save(os.path.join(output_dir, f'Y_{subset_name}.npy'), Y)

    print(f"  Saved {subset_name} set with {X.shape[0]} samples.")
    print(f"  Feature shape: {X.shape}")

# --- EXECUTION ---

if __name__ == '__main__':
    # Initialize lists to hold all training and testing data
    all_train_features = []
    all_train_labels = []
    
    # 1. Process Training Subsets
    X_train_full, Y_train_full = process_slakh_subset_parallel(SLAKH_ROOT, 'train')

    X_val = np.array([])
    X_test = np.array([])
    
    if X_train_full.size > 0:
        # Split the official 'train' folder into our project's Training and Validation sets
        indices = np.arange(X_train_full.shape[0])
        np.random.shuffle(indices)
        split_idx = int(TRAIN_RATIO * X_train_full.shape[0])
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train, Y_train = X_train_full[train_indices], Y_train_full[train_indices]
        X_val, Y_val = X_train_full[val_indices], Y_train_full[val_indices]
        
        save_data(X_val, Y_val, subset_name='val')

        # Combine training data
        all_train_features.append(X_train)
        all_train_labels.append(Y_train)

    # 2. Process Official 'validation' and add it to the main training set (more data is better)
    X_val_slakh, Y_val_slakh = process_slakh_subset_parallel(SLAKH_ROOT, 'validation')
    if X_val_slakh.size > 0:
        all_train_features.append(X_val_slakh)
        all_train_labels.append(Y_val_slakh)
        
    # 3. Process Official 'test' folder
    X_test, Y_test = process_slakh_subset_parallel(SLAKH_ROOT, 'test')
    if X_test.size > 0:
        save_data(X_test, Y_test, subset_name='test')

    # 4. Finalize and save the combined Training set
    X_train_final = np.concatenate(all_train_features, axis=0) if all_train_features else np.array([])
    Y_train_final = np.concatenate(all_train_labels, axis=0) if all_train_labels else np.array([])
    
    if X_train_final.size > 0:
        save_data(X_train_final, Y_train_final, subset_name='train')
        
    # 5. Create metadata file
    total_samples = 0
    try:
        total_samples = X_train_final.shape[0] + X_val.shape[0] + X_test.shape[0]
    except:
        pass

    with open(os.path.join(OUTPUT_DIR, 'metadata.txt'), 'w') as f:
        f.write("--- Slakh 2100 Data Generation Metadata ---\n")
        f.write(f"Source Directory: {SLAKH_ROOT}\n")
        f.write(f"Total Samples Generated: {total_samples}\n")
        f.write(f"Classes: {TARGET_CLASSES}\n")
        if total_samples > 0 and X_train_final.ndim == 4:
            # Note: X_train_final.shape[3] is the time dimension (width)
            f.write(f"Input Shape: (Samples, {N_MELS}, {X_train_final.shape[3]})\n")
        f.write(f"Output Shape: (Samples, {NUM_CLASSES})\n")
        f.write(f"Sample Duration: {SAMPLE_DURATION}s\n")
        f.write(f"Max Stems Mixed: {MAX_STEMS_PER_MIX}\n")

    print("\nPreprocessing complete. Ready for model training.")