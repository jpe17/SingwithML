# models/simple_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleResBlock(nn.Module):
    """Simple residual block without spectral normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)

class SimpleSourceVAE(nn.Module):
    """Simple VAE without spectral normalization for quick training"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simple encoder
        self.encoder_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels, stride in zip(config.vae_channels, config.vae_strides):
            layers = []
            layers.append(nn.Conv1d(in_channels, out_channels, 4, stride=stride, padding=1))
            layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.GELU())
            layers.append(SimpleResBlock(out_channels))
            
            self.encoder_layers.append(nn.Sequential(*layers))
            in_channels = out_channels
        
        # Latent projection
        self.to_mu = nn.Conv1d(in_channels, config.vae_latent_dim, 1)
        self.to_logvar = nn.Conv1d(in_channels, config.vae_latent_dim, 1)
        
        # Conservative initialization
        nn.init.xavier_uniform_(self.to_mu.weight, gain=0.01)
        nn.init.constant_(self.to_mu.bias, 0.0)
        nn.init.xavier_uniform_(self.to_logvar.weight, gain=0.01)
        nn.init.constant_(self.to_logvar.bias, -3.0)  # Start with small variance
        
        # Simple decoder
        self.from_latent = nn.Conv1d(config.vae_latent_dim, in_channels, 1)
        
        self.decoder_layers = nn.ModuleList()
        channels = list(reversed(config.vae_channels))
        strides = list(reversed(config.vae_strides))
        
        for i, (out_channels, stride) in enumerate(zip(channels[1:] + [1], strides)):
            layers = []
            layers.append(SimpleResBlock(channels[i]))
            layers.append(nn.ConvTranspose1d(channels[i], out_channels, 4, stride=stride, padding=1))
            
            if out_channels > 1:
                layers.append(nn.GroupNorm(8, out_channels))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())  # Final activation
            
            self.decoder_layers.append(nn.Sequential(*layers))

    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        
        # Stabilization
        mu = torch.clamp(mu, -1.0, 1.0)
        logvar = torch.clamp(logvar, -10.0, -1.0)
        
        return mu, logvar

    def decode(self, z):
        h = self.from_latent(z)
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            std = torch.clamp(std, 0.001, 0.5)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z 