# models/mel_processor.py
import torch
import torch.nn as nn
import torchaudio
import numpy as np

class MelSpectrogramProcessor:
    """Convert between audio and mel-spectrograms with proper preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        self.f_min = 0
        self.f_max = 8000
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=1.0,  # Use magnitude instead of power
            normalized=True
        )
        
        # Pre-emphasis filter
        self.preemphasis = 0.97
    
    def audio_to_mel(self, audio):
        """Convert audio to mel-spectrogram"""
        # Apply pre-emphasis
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        emphasized = torch.cat([
            audio[:, :1], 
            audio[:, 1:] - self.preemphasis * audio[:, :-1]
        ], dim=1)
        
        # Compute mel-spectrogram
        mel = self.mel_transform(emphasized)
        
        # Convert to log scale with small epsilon for numerical stability
        log_mel = torch.log(mel + 1e-5)
        
        # Normalize to [-1, 1] range
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
        
        return log_mel
    
    def mel_to_audio(self, mel_spec, vocoder_model):
        """Convert mel-spectrogram back to audio using vocoder"""
        # Denormalize mel-spectrogram
        # This would need to be implemented based on your vocoder's requirements
        return vocoder_model(mel_spec)

class SimpleMelVAE(nn.Module):
    """VAE operating on mel-spectrograms instead of raw audio"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel_processor = MelSpectrogramProcessor(config)
        
        # 2D convolutional encoder for mel-spectrograms
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened size (this would need to be computed based on input size)
        self.flatten_size = 256 * 5 * 32  # Adjust based on your mel-spec dimensions
        
        # Latent projection
        self.fc_mu = nn.Linear(self.flatten_size, config.vae_latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, config.vae_latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(config.vae_latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )
    
    def encode(self, x):
        # Convert audio to mel-spectrogram
        mel = self.mel_processor.audio_to_mel(x.squeeze(1))
        mel = mel.unsqueeze(1)  # Add channel dimension
        
        # Encode mel-spectrogram
        h = self.encoder(mel)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 5, 32)  # Reshape to feature map
        mel_recon = self.decoder(h)
        return mel_recon
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mel_recon = self.decode(z)
        return mel_recon, mu, logvar, z
