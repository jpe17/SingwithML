import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)

class SourceVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = 1
        
        for out_channels, stride in zip(config.vae_channels, config.vae_strides):
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 4, stride=stride, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(),
                ResidualBlock(out_channels),
                ResidualBlock(out_channels)
            ))
            in_channels = out_channels
        
        # Latent projection
        self.to_mu = nn.Conv1d(in_channels, config.vae_latent_dim, 1)
        self.to_logvar = nn.Conv1d(in_channels, config.vae_latent_dim, 1)
        
        # Decoder
        self.from_latent = nn.Conv1d(config.vae_latent_dim, in_channels, 1)
        
        self.decoder = nn.ModuleList()
        channels = list(reversed(config.vae_channels))
        strides = list(reversed(config.vae_strides))
        
        for i, (out_channels, stride) in enumerate(zip(channels[1:] + [1], strides)):
            self.decoder.append(nn.Sequential(
                ResidualBlock(channels[i]),
                ResidualBlock(channels[i]),
                nn.ConvTranspose1d(channels[i], out_channels, 4, stride=stride, padding=1),
                nn.GroupNorm(8, out_channels) if out_channels > 1 else nn.Identity(),
                nn.ReLU() if out_channels > 1 else nn.Tanh()
            ))
    
    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = layer(h)
        
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        h = self.from_latent(z)
        for layer in self.decoder:
            h = layer(h)
        return h
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
