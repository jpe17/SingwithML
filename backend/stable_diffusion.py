# models/stable_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class StableResBlock(nn.Module):
    """Stable residual block with gradient clipping and proper initialization"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Main blocks with conservative initialization
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        # Initialize with small weights to prevent instability
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x, time_emb):
        # Clamp inputs to prevent extreme values
        x = torch.clamp(x, -10.0, 10.0)
        time_emb = torch.clamp(time_emb, -10.0, 10.0)
        
        h = self.block1(x)
        
        # Add time embedding
        time_out = self.time_mlp(time_emb)
        time_out = torch.clamp(time_out, -5.0, 5.0)
        h += time_out[:, :, None]
        
        h = self.block2(h)
        
        # Clamp before residual connection
        h = torch.clamp(h, -5.0, 5.0)
        residual = self.residual_conv(x)
        
        return h + residual

class StableDiffusionModel(nn.Module):
    """Stable U-Net diffusion model with NaN prevention"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_dim = config.vae_latent_dim * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.vae_latent_dim),
            nn.Linear(config.vae_latent_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(config.vae_latent_dim * 2, 64, 1)
        
        # Simple U-Net with stable blocks
        self.down1 = StableResBlock(64, 128, time_dim)
        self.down2 = StableResBlock(128, 256, time_dim)
        self.down3 = StableResBlock(256, 256, time_dim)
        
        self.mid1 = StableResBlock(256, 256, time_dim)
        self.mid2 = StableResBlock(256, 256, time_dim)
        
        self.up3 = StableResBlock(512, 256, time_dim)  # 256 + 256 skip
        self.up2 = StableResBlock(384, 128, time_dim)  # 256 + 128 skip
        self.up1 = StableResBlock(192, 64, time_dim)   # 128 + 64 skip
        
        # Output projection with zero initialization
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv1d(64, config.vae_latent_dim, 1)
        
        # Zero initialize output
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
        # Initialize all other weights conservatively
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module != self.out_conv:
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, noisy_instrumental, time, vocal_condition):
        # Check for NaN inputs
        if torch.isnan(noisy_instrumental).any() or torch.isnan(vocal_condition).any():
            print("Warning: NaN detected in inputs!")
            noisy_instrumental = torch.nan_to_num(noisy_instrumental, 0.0)
            vocal_condition = torch.nan_to_num(vocal_condition, 0.0)
        
        # Clamp inputs
        noisy_instrumental = torch.clamp(noisy_instrumental, -5.0, 5.0)
        vocal_condition = torch.clamp(vocal_condition, -5.0, 5.0)
        
        # Time embedding
        time_emb = self.time_mlp(time)
        time_emb = torch.clamp(time_emb, -5.0, 5.0)
        
        # Input processing
        x = torch.cat([noisy_instrumental, vocal_condition], dim=1)
        x = self.input_proj(x)
        x = torch.clamp(x, -5.0, 5.0)
        
        # Encoder with skip connections
        skip1 = x
        x = self.down1(x, time_emb)
        x = F.avg_pool1d(x, 2)
        x = torch.clamp(x, -5.0, 5.0)
        
        skip2 = x
        x = self.down2(x, time_emb)
        x = F.avg_pool1d(x, 2)
        x = torch.clamp(x, -5.0, 5.0)
        
        skip3 = x
        x = self.down3(x, time_emb)
        x = F.avg_pool1d(x, 2)
        x = torch.clamp(x, -5.0, 5.0)
        
        # Middle
        x = self.mid1(x, time_emb)
        x = torch.clamp(x, -5.0, 5.0)
        x = self.mid2(x, time_emb)
        x = torch.clamp(x, -5.0, 5.0)
        
        # Decoder with skip connections
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        if x.shape[-1] != skip3.shape[-1]:
            x = F.interpolate(x, size=skip3.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.up3(x, time_emb)
        x = torch.clamp(x, -5.0, 5.0)
        
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        if x.shape[-1] != skip2.shape[-1]:
            x = F.interpolate(x, size=skip2.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_emb)
        x = torch.clamp(x, -5.0, 5.0)
        
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        if x.shape[-1] != skip1.shape[-1]:
            x = F.interpolate(x, size=skip1.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, time_emb)
        x = torch.clamp(x, -5.0, 5.0)
        
        # Output
        x = self.out_norm(x)
        x = F.gelu(x)
        x = self.out_conv(x)
        
        # Final clamp to prevent extreme outputs
        x = torch.clamp(x, -3.0, 3.0)
        
        # Check for NaN in output
        if torch.isnan(x).any():
            print("Warning: NaN detected in output! Replacing with zeros.")
            x = torch.nan_to_num(x, 0.0)
        
        return x 