import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

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

class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        b, c, l = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) l -> b h l d', h=self.heads), qkv)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * (c ** -0.5)
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h l d -> b (h d) l')
        return self.out(out) + x

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        
        self.attention = AttentionBlock(out_channels) if use_attention else None
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None]
        h = self.block2(h)
        
        if self.attention is not None:
            h = self.attention(h)
        
        return h + self.residual_conv(x)

class LatentDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_dim = config.vae_latent_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.vae_latent_dim),
            nn.Linear(config.vae_latent_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Condition embedding (CREPE features)
        self.condition_embed = nn.Sequential(
            nn.Linear(2, config.vae_latent_dim),  # 2 features from CREPE
            nn.SiLU(),
            nn.Linear(config.vae_latent_dim, config.vae_latent_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(config.vae_latent_dim * 2, config.unet_channels[0], 1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = config.unet_channels[0]
        
        for i, out_channels in enumerate(config.unet_channels[1:]):
            use_attention = i in config.unet_attention_levels
            self.encoder.append(nn.ModuleList([
                UNetBlock(in_channels, out_channels, time_dim, use_attention),
                nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
            ]))
            in_channels = out_channels
        
        # Middle
        self.middle = nn.ModuleList([
            UNetBlock(in_channels, in_channels, time_dim, True),
            UNetBlock(in_channels, in_channels, time_dim, True)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        channels = list(reversed(config.unet_channels))
        
        for i, out_channels in enumerate(channels[1:]):
            use_attention = (len(channels) - 2 - i) in config.unet_attention_levels
            self.decoder.append(nn.ModuleList([
                nn.ConvTranspose1d(channels[i], channels[i], 4, stride=2, padding=1),
                UNetBlock(channels[i] * 2, out_channels, time_dim, use_attention)
            ]))
        
        # Output projection
        self.output_proj = nn.Conv1d(config.unet_channels[0], config.vae_latent_dim, 1)
    
    def forward(self, x, time, condition):
        # Time embedding
        time_emb = self.time_mlp(time)
        
        # Condition embedding and concatenation
        if condition is not None:
            # Interpolate condition to match sequence length
            condition = F.interpolate(condition.transpose(-1, -2), size=x.shape[-1], mode='linear').transpose(-1, -2)
            condition_emb = self.condition_embed(condition).transpose(-1, -2)
            x = torch.cat([x, condition_emb], dim=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder
        skip_connections = []
        for block, downsample in self.encoder:
            x = block(x, time_emb)
            skip_connections.append(x)
            x = downsample(x)
        
        # Middle
        for block in self.middle:
            x = block(x, time_emb)
        
        # Decoder
        for (upsample, block), skip in zip(self.decoder, reversed(skip_connections)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_emb)
        
        return self.output_proj(x)
