#!/usr/bin/env python3
"""
Stable inference script for voice-to-instrumental diffusion model
with NaN protection and numerical stability
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import sys
import os
from pathlib import Path

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

from config import ModelConfig
from simple_vae import SimpleSourceVAE
from stable_diffusion import StableDiffusionModel

class StableInference:
    def __init__(self, checkpoint_path):
        self.config = ModelConfig()
        self.device = self._get_device()
        self.checkpoint_path = checkpoint_path
        
        # Initialize models
        self.vae = SimpleSourceVAE(self.config)
        self.diffusion = StableDiffusionModel(self.config)
        
        # Move to device
        self.vae.to(self.device)
        self.diffusion.to(self.device)
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Set to eval mode
        self.vae.eval()
        self.diffusion.eval()
        
        # Initialize noise schedule
        self.register_noise_schedule()
    
    def _get_device(self):
        """Get best available device"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def load_checkpoint(self):
        """Load model checkpoint with proper error handling"""
        try:
            print(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load VAE state
            if 'vae_state_dict' in checkpoint:
                self.vae.load_state_dict(checkpoint['vae_state_dict'])
                print("✅ VAE loaded successfully")
            else:
                print("⚠️ VAE state not found in checkpoint")
            
            # Load diffusion state
            if 'diffusion_state_dict' in checkpoint:
                self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
                print("✅ Diffusion model loaded successfully")
            else:
                print("⚠️ Diffusion state not found in checkpoint")
                
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise
    
    def register_noise_schedule(self):
        """Register noise schedule for diffusion sampling"""
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
        
        # Reverse process coefficients
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    def load_and_preprocess_audio(self, audio_path):
        """Load and preprocess audio with error handling"""
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            print(f"Loaded audio: {audio.shape}, sample rate: {sr}")
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                audio = resampler(audio)
                print(f"Resampled to {self.config.sample_rate}Hz")
            
            # Normalize audio
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)
            
            # Add batch dimension and move to device
            audio = audio.unsqueeze(0).to(self.device)
            
            return audio
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            raise
    
    def encode_to_latent(self, audio):
        """Encode audio to latent space with stability checks"""
        with torch.no_grad():
            try:
                # Encode through VAE
                recon, mu, logvar, z = self.vae(audio)
                
                # Check for NaN
                if torch.isnan(z).any():
                    print("Warning: NaN detected in latent encoding, using zeros")
                    z = torch.zeros_like(z)
                
                # Normalize latents
                z = z / (torch.std(z, dim=[1, 2], keepdim=True).clamp(min=1e-6) + 1e-8)
                z = torch.clamp(z, -3.0, 3.0)
                
                print(f"Encoded to latent: {z.shape}, range: [{z.min():.3f}, {z.max():.3f}]")
                return z
                
            except Exception as e:
                print(f"❌ Error in latent encoding: {e}")
                raise
    
    def decode_from_latent(self, z):
        """Decode latent back to audio with stability checks"""
        with torch.no_grad():
            try:
                # Check for NaN
                if torch.isnan(z).any():
                    print("Warning: NaN detected in latent, using zeros")
                    z = torch.zeros_like(z)
                
                # Clamp latents
                z = torch.clamp(z, -5.0, 5.0)
                
                # Decode through VAE
                audio = self.vae.decode(z)
                
                # Check for NaN in output
                if torch.isnan(audio).any():
                    print("Warning: NaN detected in decoded audio, using zeros")
                    audio = torch.zeros_like(audio)
                
                # Normalize output
                audio = torch.clamp(audio, -1.0, 1.0)
                
                print(f"Decoded audio: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")
                return audio
                
            except Exception as e:
                print(f"❌ Error in latent decoding: {e}")
                raise
    
    @torch.no_grad()
    def ddpm_sample(self, vocal_latent, num_inference_steps=20):
        """DDPM sampling with stability checks"""
        print(f"Starting DDPM sampling with {num_inference_steps} steps")
        
        # Initialize noise
        shape = vocal_latent.shape
        x_t = torch.randn(shape, device=self.device) * 0.5  # Reduced initial noise
        
        # Sampling timesteps
        timesteps = torch.linspace(self.config.diffusion_steps - 1, 0, num_inference_steps).long()
        
        for i, t in enumerate(timesteps):
            print(f"Sampling step {i+1}/{num_inference_steps}, timestep {t}")
            
            # Check for NaN
            if torch.isnan(x_t).any():
                print("Warning: NaN detected in x_t, reinitializing")
                x_t = torch.randn(shape, device=self.device) * 0.1
            
            # Clamp values
            x_t = torch.clamp(x_t, -5.0, 5.0)
            vocal_latent_clamped = torch.clamp(vocal_latent, -5.0, 5.0)
            
            # Predict noise
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            try:
                predicted_noise = self.diffusion(x_t, t_tensor, vocal_latent_clamped)
                
                # Check for NaN in prediction
                if torch.isnan(predicted_noise).any():
                    print("Warning: NaN in predicted noise, using zeros")
                    predicted_noise = torch.zeros_like(x_t)
                
                # Clamp prediction
                predicted_noise = torch.clamp(predicted_noise, -3.0, 3.0)
                
            except Exception as e:
                print(f"Warning: Error in diffusion prediction: {e}, using zeros")
                predicted_noise = torch.zeros_like(x_t)
            
            # Compute previous sample
            if t > 0:
                # Get coefficients
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t-1]
                beta_t = self.betas[t]
                
                # Compute x_0 prediction
                pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)
                
                # Compute previous sample mean
                pred_prev_mean = (
                    torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t) * pred_x0 +
                    torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) / (1 - alpha_t) * x_t
                )
                
                # Add noise if not final step
                if i < len(timesteps) - 1:
                    noise = torch.randn_like(x_t) * 0.1  # Reduced noise
                    variance = torch.sqrt(self.posterior_variance[t])
                    x_t = pred_prev_mean + variance * noise
                else:
                    x_t = pred_prev_mean
            else:
                # Final step - just remove noise
                alpha_t = self.alphas_cumprod[t]
                x_t = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Final clamp
            x_t = torch.clamp(x_t, -3.0, 3.0)
            
            # Check progress
            if torch.isnan(x_t).any():
                print("Warning: NaN detected after sampling step, using previous value")
                x_t = torch.randn(shape, device=self.device) * 0.1
        
        print("DDPM sampling completed")
        return x_t
    
    def generate_instrumental(self, vocal_path, output_path, num_inference_steps=20):
        """Generate instrumental from vocal"""
        print(f"🎵 Generating instrumental: {vocal_path} -> {output_path}")
        
        try:
            # Load and preprocess vocal
            vocal_audio = self.load_and_preprocess_audio(vocal_path)
            
            # Encode vocal to latent space
            vocal_latent = self.encode_to_latent(vocal_audio)
            
            # Generate instrumental latent using diffusion
            instrumental_latent = self.ddpm_sample(vocal_latent, num_inference_steps)
            
            # Decode to audio
            instrumental_audio = self.decode_from_latent(instrumental_latent)
            
            # Save output
            instrumental_audio = instrumental_audio.squeeze(0).cpu()
            torchaudio.save(output_path, instrumental_audio, self.config.sample_rate)
            
            print(f"✅ Instrumental saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating instrumental: {e}")
            raise

def main():
    # Test the stable inference
    checkpoint_path = "msldm_voice2instrumental/checkpoints/quick_model_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run quick training first: python quick_train.py")
        return
    
    # Initialize inference
    inference = StableInference(checkpoint_path)
    
    # Test with demo file
    vocal_path = "01_demucs/voice/voice_041000_smashmouthallstarlyrics.wav"
    output_path = "stable_test_output.wav"
    
    if os.path.exists(vocal_path):
        inference.generate_instrumental(vocal_path, output_path, num_inference_steps=10)
    else:
        print(f"❌ Test vocal file not found: {vocal_path}")

if __name__ == '__main__':
    main() 