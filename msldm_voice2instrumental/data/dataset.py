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
        print(f"Found {len(self.pairs)} pairs for {split}")
        
        # Split data
        self.pairs = self._split_data()
        print(f"Using {len(self.pairs)} pairs for {split} split")
        
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
        train_len = int(total_len * 0.8)  # Use hardcoded values for simplicity
        val_len = int(total_len * 0.1)
        
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
        
        # Load and preprocess audio - simple version
        vocal = self.preprocessor.load_simple(vocal_path)
        instrumental = self.preprocessor.load_simple(instrumental_path)
        
        # Segment audio to same length
        vocal, instrumental = self._segment_audio(vocal, instrumental)
        
        # Make sure they're the right shape
        if vocal.dim() == 1:
            vocal = vocal.unsqueeze(0)
        if instrumental.dim() == 1:
            instrumental = instrumental.unsqueeze(0)
        
        return {
            'vocal': vocal,
            'instrumental': instrumental,
        }

    def _segment_audio(self, vocal, instrumental):
        """Segment audio to fixed length"""
        min_len = min(vocal.shape[-1], instrumental.shape[-1])
        
        if min_len > self.config.segment_length:
            # Random crop during training, fixed crop for validation
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
