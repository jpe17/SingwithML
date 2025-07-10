import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import argparse
import os

from config import ModelConfig
from models.source_vae import SourceVAE
from models.diffusion_model import SimpleDiffusionModel
from data.preprocessing import AudioPreprocessor

class SimpleInference:
    def __init__(self, checkpoint_path: str):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model_config = checkpoint['model_config']
        
        # Initialize models
        self.vae = SourceVAE(self.model_config)
        self.diffusion = SimpleDiffusionModel(self.model_config)
        
        # Load state dicts
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f"Inference using device: {self.device}")
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
        ).to(self.device)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]).to(self.device)
        
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(self.device)
        
        # Calculate posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(self.device)
    
    @torch.no_grad()
    def ddpm_sample(self, shape, vocal_condition):
        """DDPM sampling process"""
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Create time steps for inference
        timesteps = torch.linspace(self.model_config.diffusion_steps - 1, 0, self.model_config.inference_steps).long()
        
        # Reverse diffusion process
        for i, t_idx in enumerate(tqdm(timesteps, desc='Sampling')):
            t = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.diffusion(x, t, vocal_condition)
            
            # Get noise schedule values
            alpha_t = self.alphas_cumprod[t][:, None, None]
            alpha_t_prev = self.alphas_cumprod_prev[t][:, None, None] if t_idx > 0 else torch.ones_like(alpha_t)
            beta_t = self.betas[t][:, None, None]
            
            # Predict x_0 (clean latent)
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            
            # Compute direction pointing to x_t
            if t_idx > 0:
                # Add noise for intermediate steps
                noise = torch.randn_like(x)
                # DDPM formula
                x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * pred_noise
            else:
                # Final step - no noise
                x = pred_x0
        
        return x

    @torch.no_grad()
    def generate_instrumental(self, vocal_path: str, output_path: str):
        """Generate instrumental from vocal"""
        print(f"Loading vocal: {vocal_path}")
        
        # Load and preprocess vocal
        vocal = self.preprocessor.load_simple(vocal_path)
        vocal = vocal.unsqueeze(0).unsqueeze(0).to(self.device)  # Add batch and channel dims
        
        # Trim to model's expected length
        max_length = self.model_config.segment_length
        if vocal.shape[-1] > max_length:
            vocal = vocal[..., :max_length]
            print(f"Trimmed audio to {max_length} samples ({max_length/self.model_config.sample_rate:.2f} seconds)")
        elif vocal.shape[-1] < max_length:
            # Pad if too short
            padding = max_length - vocal.shape[-1]
            vocal = torch.nn.functional.pad(vocal, (0, padding))
            print(f"Padded audio to {max_length} samples")
        
        # Encode vocal to latent space
        print("Encoding vocal to latent space...")
        _, _, _, vocal_z = self.vae(vocal)
        print(f"Vocal latent shape: {vocal_z.shape}")
        
        # Generate instrumental latent
        print("Generating instrumental latent...")
        latent_shape = vocal_z.shape
        instrumental_latent = self.ddpm_sample(latent_shape, vocal_z)
        
        # Decode to audio
        print("Decoding to audio...")
        instrumental = self.vae.decode(instrumental_latent)
        
        # Prepare output path
        if os.path.isdir(output_path):
            vocal_filename = os.path.basename(vocal_path)
            name_without_ext = os.path.splitext(vocal_filename)[0]
            output_filename = f"{name_without_ext}_instrumental.wav"
            output_path = os.path.join(output_path, output_filename)
        elif not output_path.endswith(('.wav', '.mp3', '.flac')):
            output_path = output_path + '.wav'
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save output
        instrumental = instrumental.squeeze(0).cpu()
        torchaudio.save(output_path, instrumental, self.model_config.sample_rate)
        
        print(f"Generated instrumental saved to: {output_path}")
        print(f"Audio length: {instrumental.shape[-1]/self.model_config.sample_rate:.2f} seconds")
    
    def test_vae_reconstruction(self, vocal_path: str, output_dir: str):
        """Test VAE reconstruction quality"""
        print("Testing VAE reconstruction...")
        
        # Load audio
        vocal = self.preprocessor.load_simple(vocal_path)
        vocal = vocal.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Trim/pad to expected length
        max_length = self.model_config.segment_length
        if vocal.shape[-1] > max_length:
            vocal = vocal[..., :max_length]
        elif vocal.shape[-1] < max_length:
            padding = max_length - vocal.shape[-1]
            vocal = torch.nn.functional.pad(vocal, (0, padding))
        
        # Test VAE reconstruction
        with torch.no_grad():
            mu, logvar = self.vae.encode(vocal)
            z = self.vae.reparameterize(mu, logvar)
            reconstructed = self.vae.decode(z)
        
        # Calculate reconstruction error
        recon_loss = torch.nn.functional.mse_loss(reconstructed, vocal)
        print(f"VAE reconstruction loss: {recon_loss.item():.6f}")
        
        # Save reconstruction
        reconstructed = reconstructed.squeeze(0).cpu()
        
        os.makedirs(output_dir, exist_ok=True)
        vocal_filename = os.path.basename(vocal_path)
        name_without_ext = os.path.splitext(vocal_filename)[0]
        recon_path = os.path.join(output_dir, f"{name_without_ext}_vae_reconstruction.wav")
        
        torchaudio.save(recon_path, reconstructed, self.model_config.sample_rate)
        print(f"VAE reconstruction saved to: {recon_path}")
        
        return recon_loss.item()

def main():
    parser = argparse.ArgumentParser(description='Simple Voice-to-Instrumental Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--vocal', type=str, required=True, help='Path to vocal file')
    parser.add_argument('--output', type=str, required=True, help='Output path (file or directory)')
    parser.add_argument('--test-vae', action='store_true', help='Test VAE reconstruction quality first')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = SimpleInference(args.checkpoint)
    
    # Test VAE reconstruction if requested
    if args.test_vae:
        output_dir = args.output if os.path.isdir(args.output) else os.path.dirname(args.output)
        recon_loss = inference.test_vae_reconstruction(args.vocal, output_dir)
        if recon_loss > 0.1:
            print(f"WARNING: High VAE reconstruction loss ({recon_loss:.6f}). Model needs more training.")
    
    # Generate instrumental
    inference.generate_instrumental(args.vocal, args.output)

if __name__ == '__main__':
    main() 