import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F 
import torchaudio 

from config import ModelConfig, DataConfig
from data.dataset import VoiceInstrumentalDataset
from msldm_voice2instrumental.models.enhanced_vae import SourceVAE
from models.diffusion_model import SimpleDiffusionModel

class SimpleTrainer:
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        self.model_config = model_config
        self.data_config = data_config
        
        # Initialize models
        self.vae = SourceVAE(model_config)
        self.diffusion = SimpleDiffusionModel(model_config)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")
        self.vae.to(self.device)
        self.diffusion.to(self.device)
        
        # Optimizers
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=model_config.learning_rate)
        self.diffusion_optimizer = optim.Adam(self.diffusion.parameters(), lr=model_config.learning_rate)
        
        # Noise schedule
        self.register_noise_schedule()
        
        # Datasets
        try:
            self.train_dataset = VoiceInstrumentalDataset(
                data_config.vocals_folder, 
                data_config.instruments_folder, 
                model_config,
                'train'
            )
            self.val_dataset = VoiceInstrumentalDataset(
                data_config.vocals_folder, 
                data_config.instruments_folder, 
                model_config,
                'val'
            )
        except Exception as e:
            print(f"Error creating datasets: {e}")
            raise
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=model_config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=model_config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
    def register_noise_schedule(self):
        """Register noise schedule for diffusion"""
        betas = torch.linspace(
            self.model_config.beta_start, 
            self.model_config.beta_end, 
            self.model_config.diffusion_steps
        ).to(self.device)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        
    def train_vae_step(self, vocal, instrumental, epoch, step, total_steps):
        """Enhanced VAE training with spectral loss, cyclical KL annealing,
        and fixed-length cropping to avoid size mismatches."""
        # Zero gradients
        self.vae_optimizer.zero_grad()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_spectral_loss = 0.0

        # Iterate over both vocal and instrumental inputs
        for audio in (vocal, instrumental):
            # Forward pass through VAE
            recon, mu, logvar, z = self.vae(audio)

            # --- Crop recon and audio to the same minimum length ---
            min_len = min(recon.shape[-1], audio.shape[-1])
            recon = recon[..., :min_len]
            audio = audio[..., :min_len]

            # Normalize latents for diffusion compatibility
            z_normalized = self.normalize_latents(z)

            # Reconstruction losses
            mse_loss = F.mse_loss(recon, audio)
            spec_loss = self.spectral_loss(recon, audio)
            recon_loss = mse_loss + 0.1 * spec_loss

            # KL divergence loss (per‐sample normalized)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl = kl / audio.numel()

            # Cyclical KL weight schedule
            kl_weight = self.get_kl_weight(epoch, step, total_steps)

            # Combine losses
            loss = recon_loss + kl_weight * kl
            total_loss += loss
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl.item()
            total_spectral_loss += spec_loss.item()

        # Backpropagation and optimization step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.model_config.gradient_clip)
        self.vae_optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': total_recon_loss,
            'kl_loss': total_kl_loss,
            'spectral_loss': total_spectral_loss,
            'kl_weight': kl_weight
        }

    
    def train_diffusion_step(self, vocal, instrumental):
        """Train diffusion model"""
        self.diffusion_optimizer.zero_grad()
        
        # Encode both to latent space
        with torch.no_grad():
            _, _, _, vocal_z = self.vae(vocal)
            _, _, _, instrumental_z = self.vae(instrumental)
        
        # Random timesteps
        t = torch.randint(0, self.model_config.diffusion_steps, (vocal_z.shape[0],), device=self.device)
        
        # Add noise to instrumental latents
        noise = torch.randn_like(instrumental_z)
        noisy_instrumental = self.q_sample(instrumental_z, t, noise)
        
        # Predict noise given vocal conditioning
        pred_noise = self.diffusion(noisy_instrumental, t, vocal_z)
        
        # Loss
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.model_config.gradient_clip)
        self.diffusion_optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.vae.train()
        self.diffusion.train()
        
        vae_losses = []
        diffusion_losses = []
        
        for i, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}')):
            vocal = batch['vocal'].to(self.device)
            instrumental = batch['instrumental'].to(self.device)
            
            # Train VAE
            vae_loss = self.train_vae_step(vocal, instrumental)
            vae_losses.append(vae_loss)
            
            # Train diffusion (only if VAE is somewhat trained)
            if epoch >= 5:  # Start diffusion training after a few VAE epochs
                diffusion_loss = self.train_diffusion_step(vocal, instrumental)
                diffusion_losses.append(diffusion_loss)
            
            # Log every 10 batches
            if i % 10 == 0:
                print(f"Batch {i}: VAE Loss: {vae_loss:.6f}", end="")
                if epoch >= 5 and diffusion_losses:
                    print(f", Diffusion Loss: {diffusion_losses[-1]:.6f}")
                else:
                    print("")
        
        avg_vae_loss = np.mean(vae_losses)
        avg_diffusion_loss = np.mean(diffusion_losses) if diffusion_losses else 0
        
        print(f"Epoch {epoch+1} - VAE Loss: {avg_vae_loss:.6f}, Diffusion Loss: {avg_diffusion_loss:.6f}")
        
        return avg_vae_loss, avg_diffusion_loss
    
    def validate(self):
        """Quick validation"""
        self.vae.eval()
        self.diffusion.eval()
        
        vae_losses = []
        diffusion_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                vocal = batch['vocal'].to(self.device)
                instrumental = batch['instrumental'].to(self.device)
                
                # VAE validation
                recon, mu, logvar, z = self.vae(instrumental)
                recon_loss = F.mse_loss(recon, instrumental)
                vae_losses.append(recon_loss.item())
                
                # Diffusion validation
                _, _, _, vocal_z = self.vae(vocal)
                _, _, _, instrumental_z = self.vae(instrumental)
                
                t = torch.randint(0, self.model_config.diffusion_steps, (vocal_z.shape[0],), device=self.device)
                noise = torch.randn_like(instrumental_z)
                noisy_instrumental = self.q_sample(instrumental_z, t, noise)
                pred_noise = self.diffusion(noisy_instrumental, t, vocal_z)
                diff_loss = F.mse_loss(pred_noise, noise)
                diffusion_losses.append(diff_loss.item())
        
        return np.mean(vae_losses), np.mean(diffusion_losses)
    
    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'model_config': self.model_config,
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/simple_model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting simplified training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.model_config.num_epochs):
            # Training
            train_vae_loss, train_diff_loss = self.train_epoch(epoch)
            
            # Validation
            val_vae_loss, val_diff_loss = self.validate()
            print(f"Validation - VAE: {val_vae_loss:.6f}, Diffusion: {val_diff_loss:.6f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            current_val_loss = val_vae_loss + val_diff_loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.save_checkpoint(f"best")
                print(f"New best model saved!")
        
        print("Training completed!")

def main():
    model_config = ModelConfig()
    data_config = DataConfig(
        vocals_folder='/Users/joaoesteves/mli/SingwithML/03_data_preprocessing/voice',      
        instruments_folder='/Users/joaoesteves/mli/SingwithML/03_data_preprocessing/instrumental'  
    )
    
    trainer = SimpleTrainer(model_config, data_config)
    trainer.train()

if __name__ == '__main__':
    main() 