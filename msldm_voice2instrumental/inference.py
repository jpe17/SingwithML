import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import argparse
import os

from config import ModelConfig, DataConfig
from models.source_vae import SourceVAE
from models.diffusion_model import LatentDiffusionModel
from data.preprocessing import AudioPreprocessor

class MSLDMInference:
    def __init__(self, checkpoint_path: str):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model_config = checkpoint['model_config']
        
        # Initialize models
        self.vae = SourceVAE(self.model_config)
        self.diffusion = LatentDiffusionModel(self.model_config)
        
        # Load state dicts
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        
        # Move to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae.to(self.device)
        self.diffusion.to(self.device)
        
        # Set to eval mode
        self.vae.eval()
        self.diffusion.eval()
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(self.model_config)
        
        # Initialize noise schedule
        self.register_noise_schedule()
    
    def register_noise_schedule(self):
        """Register noise schedule for inference"""
        betas = torch.linspace(
            self.model_config.beta_start, 
            self.model_config.beta_end, 
            self.model_config.diffusion_steps
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        self.betas = betas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(self.device)
        
        # Calculate posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(self.device)
    
    @torch.no_grad()
    def ddpm_sample(self, shape, condition):
        """DDPM sampling"""
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Reverse diffusion process
        for i in tqdm(reversed(range(self.model_config.inference_steps)), desc='Sampling'):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            # Scale t to full range
            t_scaled = t * (self.model_config.diffusion_steps // self.model_config.inference_steps)
            
            # Predict noise
            pred_noise = self.diffusion(x, t_scaled, condition)
            
            # Compute denoised sample
            alpha = self.alphas_cumprod[t_scaled][:, None, None]
            alpha_prev = self.alphas_cumprod_prev[t_scaled][:, None, None]
            beta = self.betas[t_scaled][:, None, None]
            
            # Compute x_{t-1}
            pred_x0 = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
                x = x + torch.sqrt(beta) * noise
            else:
                x = pred_x0
        
        return x
    
def generate_instrumental(self, vocal_path: str, output_path: str, microphone_mode: bool = True):
    """Generate instrumental from vocal"""
    if microphone_mode:
        # Simple enhanced preprocessing for microphone
        vocal = self.preprocessor.preprocess_microphone_audio(vocal_path)
        print("Applied simple microphone preprocessing")
    else:
        # Basic preprocessing for clean audio
        vocal = self.preprocessor.load_and_preprocess(vocal_path, skip_preprocessing=False)
        print("Applied basic preprocessing")
    
    vocal = vocal.unsqueeze(0).to(self.device)
    
    # Extract CREPE features
    pitch_features = self.preprocessor.extract_crepe_features(vocal.squeeze(0))
    pitch_features = pitch_features.unsqueeze(0).to(self.device)
    
    # Generate latent
    latent_shape = (1, self.model_config.vae_latent_dim, vocal.shape[-1] // 8)
    latent = self.ddmp_sample(latent_shape, pitch_features)
    
    # Decode to audio
    with torch.no_grad():
        instrumental = self.vae.decode(latent)
    
    # Save output
    instrumental = instrumental.squeeze(0).cpu()
    torchaudio.save(output_path, instrumental, self.model_config.sample_rate)
    
    print(f"Generated instrumental saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='MSLDM Voice-to-Instrumental Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--vocal', type=str, required=True, help='Path to vocal file')
    parser.add_argument('--output', type=str, required=True, help='Output instrumental file path')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = MSLDMInference(args.checkpoint)
    
    # Generate instrumental
    inference.generate_instrumental(args.vocal, args.output)

if __name__ == '__main__':
    main()
