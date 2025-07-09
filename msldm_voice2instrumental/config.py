from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    # Audio parameters
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    
    # VAE parameters
    vae_latent_dim: int = 80
    vae_channels: List[int] = None
    vae_strides: List[int] = None
    
    # Diffusion parameters
    diffusion_steps: int = 1000
    inference_steps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # U-Net parameters
    unet_channels: List[int] = None
    unet_attention_levels: List[int] = None
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_clip: float = 1.0
    
    # Data parameters
    segment_length: int = 131072  # ~3 seconds at 44.1kHz
    
    def __post_init__(self):
        if self.vae_channels is None:
            self.vae_channels = [64, 128, 256, 512]
        if self.vae_strides is None:
            self.vae_strides = [2, 2, 2, 2]
        if self.unet_channels is None:
            self.unet_channels = [128, 256, 512, 1024]
        if self.unet_attention_levels is None:
            self.unet_attention_levels = [1, 2, 3]

@dataclass
class DataConfig:
    vocals_folder: str = "vocals"
    instruments_folder: str = "instruments"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Augmentation parameters
    noise_prob: float = 0.3
    noise_snr_range: tuple = (0.1, 0.3)
    pitch_shift_range: tuple = (-2, 2)
    time_stretch_range: tuple = (0.8, 1.2)
    volume_range: tuple = (-6, 6)
