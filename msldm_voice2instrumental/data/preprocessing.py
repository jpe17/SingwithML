import torch
import torchaudio
import librosa
import numpy as np
import crepe
from typing import Tuple, Optional
import scipy.signal

class AudioPreprocessor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        
    def load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Load and apply basic preprocessing"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to 44.1kHz
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # High-pass filter at 80Hz
        waveform = self.high_pass_filter(waveform, 80)
        
        # Loudness normalization to -20 LUFS
        waveform = self.lufs_normalize(waveform, -20)
        
        return waveform
    
    def high_pass_filter(self, waveform: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Apply high-pass filter"""
        from scipy.signal import butter, sosfilt
        
        sos = butter(6, cutoff, btype='high', fs=self.sample_rate, output='sos')
        filtered = torch.from_numpy(
            sosfilt(sos, waveform.numpy())
        ).float()
        return filtered

    
    def lufs_normalize(self, waveform: torch.Tensor, target_lufs: float) -> torch.Tensor:
        """Normalize to target LUFS (simplified implementation)"""
        # Simplified LUFS normalization
        rms = torch.sqrt(torch.mean(waveform**2))
        target_rms = 10**(target_lufs / 20)
        scaling_factor = target_rms / (rms + 1e-8)
        return waveform * scaling_factor
    
    def extract_crepe_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract CREPE pitch features"""
        audio_np = waveform.squeeze().numpy()
        
        # CREPE pitch extraction
        time, frequency, confidence, _ = crepe.predict(
            audio_np, 
            sr=self.sample_rate,
            model_capacity='large',
            viterbi=True,
            step_size=self.config.hop_length/self.sample_rate*1000
        )
        
        # Convert to features
        pitch_features = np.stack([frequency, confidence], axis=1)
        return torch.from_numpy(pitch_features).float()
    
    def apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply training augmentations"""
        # Gaussian noise
        if np.random.random() < self.config.noise_prob:
            snr = np.random.uniform(*self.config.noise_snr_range)
            noise = torch.randn_like(waveform) * snr
            waveform = waveform + noise
        
        # Pitch shifting
        if np.random.random() < 0.5:
            shift_semitones = np.random.uniform(*self.config.pitch_shift_range)
            waveform = self.pitch_shift(waveform, shift_semitones)
        
        # Time stretching
        if np.random.random() < 0.5:
            stretch_factor = np.random.uniform(*self.config.time_stretch_range)
            waveform = self.time_stretch(waveform, stretch_factor)
        
        # Volume adjustment
        if np.random.random() < 0.5:
            gain_db = np.random.uniform(*self.config.volume_range)
            gain_linear = 10**(gain_db / 20)
            waveform = waveform * gain_linear
        
        return waveform
    
    def pitch_shift(self, waveform: torch.Tensor, semitones: float) -> torch.Tensor:
        """Pitch shift using librosa"""
        shifted = librosa.effects.pitch_shift(
            waveform.numpy(), 
            sr=self.sample_rate, 
            n_steps=semitones
        )
        return torch.from_numpy(shifted).float()
    
    def time_stretch(self, waveform: torch.Tensor, rate: float) -> torch.Tensor:
        """Time stretch using librosa"""
        stretched = librosa.effects.time_stretch(
            waveform.numpy(), 
            rate=rate
        )
        return torch.from_numpy(stretched).float()
