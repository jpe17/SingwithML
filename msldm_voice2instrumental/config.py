# config.py
class ModelConfig:
    def __init__(self):
        # Audio settings
        self.sample_rate = 22050
        self.segment_length = 65536  # ~3 seconds at 22050 Hz
        
        # VAE Architecture
        self.vae_channels = [64, 128, 256, 512]
        self.vae_strides = [2, 4, 4, 4]
        self.vae_latent_dim = 128
        
        # Diffusion settings
        self.diffusion_steps = 1000
        self.inference_steps = 50
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # U-Net channels
        self.unet_channels = [128, 256, 512, 1024]
        
        # Training settings
        self.batch_size = 8  # Reduced for memory efficiency
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.gradient_clip = 1.0
        
        # Audio preprocessing
        self.hop_length = 256
        self.noise_prob = 0.3
        self.noise_snr_range = (0.01, 0.05)
        self.pitch_shift_range = (-2, 2)
        self.time_stretch_range = (0.95, 1.05)
        self.volume_range = (-5, 5)

class DataConfig:
    def __init__(self, vocals_folder, instruments_folder):
        self.vocals_folder = vocals_folder
        self.instruments_folder = instruments_folder
