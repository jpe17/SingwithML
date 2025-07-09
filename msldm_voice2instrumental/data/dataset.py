import torch
from torch.utils.data import Dataset
import os
import glob
from .preprocessing import AudioPreprocessor

class VoiceInstrumentalDataset(Dataset):
    def __init__(self, vocals_folder: str, instruments_folder: str, config, split='train'):
        self.vocals_folder = vocals_folder
        self.instruments_folder = instruments_folder
        self.config = config
        self.split = split
        self.preprocessor = AudioPreprocessor(config)
        
        # Find matching pairs
        self.pairs = self._find_pairs()
        
        # Split data
        self.pairs = self._split_data()
        
    def _find_pairs(self):
        """Find matching vocal and instrumental files"""
        pairs = []
        
        vocal_files = glob.glob(os.path.join(self.vocals_folder, "voice_*.wav"))
        
        for vocal_file in vocal_files:
            # Extract base name
            base_name = os.path.basename(vocal_file).replace("voice_", "")
            
            # Find corresponding instrumental file
            instrumental_file = os.path.join(
                self.instruments_folder, 
                f"instrumental_{base_name}"
            )
            
            if os.path.exists(instrumental_file):
                pairs.append((vocal_file, instrumental_file))
        
        return pairs
    
    def _split_data(self):
        """Split data into train/val/test"""
        total_len = len(self.pairs)
        train_len = int(total_len * self.config.train_split)
        val_len = int(total_len * self.config.val_split)
        
        if self.split == 'train':
            return self.pairs[:train_len]
        elif self.split == 'val':
            return self.pairs[train_len:train_len+val_len]
        else:  # test
            return self.pairs[train_len+val_len:]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        vocal_path, instrumental_path = self.pairs[idx]
        
        print(f"Processing file {idx+1}/{len(self.pairs)}: {os.path.basename(vocal_path)}")
        
        # Load and preprocess audio
        print("  Loading vocal...")
        vocal = self.preprocessor.load_and_preprocess(vocal_path)
        print("  Loading instrumental...")
        instrumental = self.preprocessor.load_and_preprocess(instrumental_path)
        
        # Apply augmentations during training
        if self.split == 'train':
            print("  Applying augmentations...")
            vocal = self.preprocessor.apply_augmentations(vocal)
            instrumental = self.preprocessor.apply_augmentations(instrumental)
        
        # Segment audio
        print("  Segmenting audio...")
        vocal, instrumental = self._segment_audio(vocal, instrumental)
        
        # Extract CREPE features
        print("  Extracting CREPE features...")
        pitch_features = self.preprocessor.extract_crepe_features(vocal)
        print("  Done!")
        
        return {
            'vocal': vocal,
            'instrumental': instrumental,
            'pitch_features': pitch_features
        }

    
    def _segment_audio(self, vocal, instrumental):
        """Segment audio to fixed length"""
        min_len = min(vocal.shape[-1], instrumental.shape[-1])
        
        if min_len > self.config.segment_length:
            # Random crop during training
            if self.split == 'train':
                start = torch.randint(0, min_len - self.config.segment_length, (1,)).item()
            else:
                start = 0
            
            vocal = vocal[..., start:start + self.config.segment_length]
            instrumental = instrumental[..., start:start + self.config.segment_length]
        else:
            # Pad if too short
            pad_len = self.config.segment_length - min_len
            vocal = torch.nn.functional.pad(vocal, (0, pad_len))
            instrumental = torch.nn.functional.pad(instrumental, (0, pad_len))
        
        return vocal, instrumental
