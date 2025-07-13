#!/usr/bin/env python3
"""
Quick training script for voice-to-instrumental diffusion model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import os
import numpy as np
from tqdm import tqdm
import sys

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

from config import ModelConfig
from simple_vae import SimpleSourceVAE
from stable_diffusion import StableDiffusionModel

class QuickTrainer:
    def __init__(self):
        self.config = ModelConfig()
        
        # Reduce parameters for quick training
        self.config.num_epochs = 20
        self.config.batch_size = 2
        self.config.learning_rate = 1e-4
        self.config.segment_length = 65536  # ~3 seconds at 22kHz
        
        # Initialize models
        self.vae = SimpleSourceVAE(self.config)
        self.diffusion = StableDiffusionModel(self.config)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"Training on device: {self.device}")
        
        self.vae.to(self.device)
        self.diffusion.to(self.device)
        
        # Optimizers
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=self.config.learning_rate)
        self.diffusion_optimizer = optim.Adam(self.diffusion.parameters(), lr=self.config.learning_rate)
        
        # Noise schedule
        self.register_noise_schedule()
        
        # Load data
        self.load_data()
    
    def register_noise_schedule(self):
        """Register noise schedule for diffusion"""
        betas = torch.linspace(
            self.config.beta_start, 
            self.config.beta_end, 
            self.config.diffusion_steps
        ).to(self.device)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    def load_data(self):
        """Load a subset of the data for quick training"""
        vocal_dir = "01_demucs/voice"
        instrumental_dir = "01_demucs/instrumental"
        
        # Get matching pairs
        vocal_files = sorted([f for f in os.listdir(vocal_dir) if f.endswith('.wav')])
        instrumental_files = sorted([f for f in os.listdir(instrumental_dir) if f.endswith('.wav')])
        
        # Take only first 10 files for quick training
        self.data_pairs = []
        for i in range(min(10, len(vocal_files), len(instrumental_files))):
            vocal_path = os.path.join(vocal_dir, vocal_files[i])
            instrumental_path = os.path.join(instrumental_dir, instrumental_files[i])
            
            if os.path.exists(vocal_path) and os.path.exists(instrumental_path):
                self.data_pairs.append((vocal_path, instrumental_path))
        
        print(f"Loaded {len(self.data_pairs)} training pairs")
    
    def load_audio(self, path):
        """Load and preprocess audio"""
        audio, sr = torchaudio.load(path)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            audio = resampler(audio)
        
        # Trim or pad to segment length
        if audio.shape[1] > self.config.segment_length:
            start = np.random.randint(0, audio.shape[1] - self.config.segment_length)
            audio = audio[:, start:start + self.config.segment_length]
        elif audio.shape[1] < self.config.segment_length:
            padding = self.config.segment_length - audio.shape[1]
            audio = F.pad(audio, (0, padding))
        
        return audio
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def normalize_latents(self, z):
        """Normalize latents for diffusion compatibility"""
        return z / torch.std(z, dim=[1, 2], keepdim=True).clamp(min=1e-6)
    
    def spectral_loss(self, recon, target):
        """Simple spectral loss"""
        # Convert to frequency domain
        recon_fft = torch.fft.rfft(recon, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Magnitude loss
        recon_mag = torch.abs(recon_fft)
        target_mag = torch.abs(target_fft)
        
        return F.mse_loss(recon_mag, target_mag)
    
    def train_vae_step(self, vocal, instrumental):
        """Train VAE with NaN protection"""
        self.vae_optimizer.zero_grad()
        
        total_loss = 0
        
        # Train on both vocal and instrumental
        for audio in [vocal, instrumental]:
            # Check input for NaN
            if torch.isnan(audio).any():
                print("Warning: NaN in input audio, skipping")
                continue
                
            recon, mu, logvar, z = self.vae(audio)
            
            # Check VAE outputs for NaN
            if torch.isnan(recon).any() or torch.isnan(mu).any() or torch.isnan(logvar).any():
                print("Warning: NaN in VAE outputs, skipping")
                continue
            
            # Ensure same length
            min_len = min(recon.shape[-1], audio.shape[-1])
            recon = recon[..., :min_len]
            audio = audio[..., :min_len]
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, audio)
            
            # Spectral loss
            spec_loss = self.spectral_loss(recon, audio)
            
            # KL loss with clamping
            logvar = torch.clamp(logvar, -10.0, 10.0)
            mu = torch.clamp(mu, -5.0, 5.0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / audio.numel()
            
            # Check losses for NaN
            if torch.isnan(recon_loss) or torch.isnan(spec_loss) or torch.isnan(kl_loss):
                print("Warning: NaN in loss components, skipping")
                continue
            
            # Combined loss
            loss = recon_loss + 0.1 * spec_loss + 0.1 * kl_loss
            total_loss += loss
        
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 0.5)
            
            # Check gradients
            has_nan_grad = False
            for param in self.vae.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if not has_nan_grad:
                self.vae_optimizer.step()
            else:
                print("Warning: NaN gradients in VAE, skipping optimizer step")
        
        return total_loss.item() if total_loss > 0 else 0.0
    
    def train_diffusion_step(self, vocal, instrumental):
        """Train diffusion model with NaN protection"""
        self.diffusion_optimizer.zero_grad()
        
        # Encode to latent space
        with torch.no_grad():
            _, _, _, vocal_z = self.vae(vocal)
            _, _, _, instrumental_z = self.vae(instrumental)
            
            # Check for NaN in latents
            if torch.isnan(vocal_z).any() or torch.isnan(instrumental_z).any():
                print("Warning: NaN detected in VAE latents, skipping batch")
                return 0.0
            
            # Normalize latents to prevent extreme values
            vocal_z = self.normalize_latents(vocal_z)
            instrumental_z = self.normalize_latents(instrumental_z)
        
        # Random timesteps
        t = torch.randint(0, self.config.diffusion_steps, (vocal_z.shape[0],), device=self.device)
        
        # Add noise with controlled variance
        noise = torch.randn_like(instrumental_z) * 0.5  # Reduce noise variance
        noisy_instrumental = self.q_sample(instrumental_z, t, noise)
        
        # Check for NaN before forward pass
        if torch.isnan(noisy_instrumental).any():
            print("Warning: NaN in noisy_instrumental, skipping batch")
            return 0.0
        
        # Predict noise
        pred_noise = self.diffusion(noisy_instrumental, t, vocal_z)
        
        # Check for NaN in prediction
        if torch.isnan(pred_noise).any():
            print("Warning: NaN in diffusion prediction, skipping batch")
            return 0.0
        
        # Loss with stability check
        loss = F.mse_loss(pred_noise, noise)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected, skipping batch")
            return 0.0
        
        loss.backward()
        
        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 0.5)
        
        # Check gradients for NaN
        has_nan_grad = False
        for param in self.diffusion.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("Warning: NaN gradients detected, skipping optimizer step")
            return 0.0
        
        self.diffusion_optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.vae.train()
        self.diffusion.train()
        
        vae_losses = []
        diffusion_losses = []
        
        # Create batches
        np.random.shuffle(self.data_pairs)
        
        for i in range(0, len(self.data_pairs), self.config.batch_size):
            batch_pairs = self.data_pairs[i:i + self.config.batch_size]
            
            # Load batch
            vocals = []
            instrumentals = []
            
            for vocal_path, instrumental_path in batch_pairs:
                vocal = self.load_audio(vocal_path)
                instrumental = self.load_audio(instrumental_path)
                vocals.append(vocal)
                instrumentals.append(instrumental)
            
            vocal_batch = torch.stack(vocals).to(self.device)
            instrumental_batch = torch.stack(instrumentals).to(self.device)
            
            # Train VAE
            vae_loss = self.train_vae_step(vocal_batch, instrumental_batch)
            vae_losses.append(vae_loss)
            
            # Train diffusion (after a few VAE epochs)
            if epoch >= 3:
                diffusion_loss = self.train_diffusion_step(vocal_batch, instrumental_batch)
                diffusion_losses.append(diffusion_loss)
            
            print(f"Epoch {epoch+1}, Batch {i//self.config.batch_size + 1}: VAE={vae_loss:.4f}", end="")
            if epoch >= 3 and diffusion_losses:
                print(f", Diffusion={diffusion_losses[-1]:.4f}")
            else:
                print("")
        
        avg_vae = np.mean(vae_losses)
        avg_diffusion = np.mean(diffusion_losses) if diffusion_losses else 0
        
        print(f"Epoch {epoch+1} Summary: VAE={avg_vae:.4f}, Diffusion={avg_diffusion:.4f}")
        
        return avg_vae, avg_diffusion
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'model_config': self.config,
        }
        
        os.makedirs('msldm_voice2instrumental/checkpoints', exist_ok=True)
        
        if is_best:
            path = 'msldm_voice2instrumental/checkpoints/quick_model_best.pt'
        else:
            path = f'msldm_voice2instrumental/checkpoints/quick_model_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop"""
        print("🎵 Starting Quick Training 🎵")
        print(f"Device: {self.device}")
        print(f"Data pairs: {len(self.data_pairs)}")
        print(f"Epochs: {self.config.num_epochs}")
        print()
        
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            vae_loss, diffusion_loss = self.train_epoch(epoch)
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            total_loss = vae_loss + diffusion_loss
            if total_loss < best_loss:
                best_loss = total_loss
                self.save_checkpoint(epoch + 1, is_best=True)
                print("🎯 New best model saved!")
            
            print()
        
        print("✅ Quick training completed!")
        print(f"Best model saved at: msldm_voice2instrumental/checkpoints/quick_model_best.pt")

def main():
    trainer = QuickTrainer()
    trainer.train()

if __name__ == '__main__':
    main() 