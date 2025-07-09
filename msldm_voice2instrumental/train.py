import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import wandb
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F 

from config import ModelConfig, DataConfig
from data.dataset import VoiceInstrumentalDataset
from models.source_vae import SourceVAE
from models.diffusion_model import LatentDiffusionModel

class MSLDMTrainer:
    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        self.model_config = model_config
        self.data_config = data_config
        
        # Initialize models
        self.vae = SourceVAE(model_config)
        self.diffusion = LatentDiffusionModel(model_config)
        
        # Move to GPU
        # Current device selection (around line 35)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Updated with MPS support
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
        
        # Schedulers
        self.vae_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.vae_optimizer, T_max=model_config.num_epochs
        )
        self.diffusion_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.diffusion_optimizer, T_max=model_config.num_epochs
        )
        
        # Noise schedule
        self.register_noise_schedule()
        
        # Datasets
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
        
    def register_noise_schedule(self):
        """Register noise schedule for diffusion"""
        betas = torch.linspace(
            self.model_config.beta_start, 
            self.model_config.beta_end, 
            self.model_config.diffusion_steps
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        """Register buffer for noise schedule"""
        setattr(self, name, tensor.to(self.device))
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def train_vae_step(self, batch):
        """Training step for VAE"""
        self.vae_optimizer.zero_grad()
        
        instrumental = batch['instrumental'].to(self.device)
        recon, mu, logvar, z = self.vae(instrumental)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, instrumental)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / instrumental.numel()
        
        # Total loss
        loss = recon_loss + 0.001 * kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.model_config.gradient_clip)
        self.vae_optimizer.step()
        
        return {
            'vae_loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train_diffusion_step(self, batch):
        """Training step for diffusion model"""
        self.diffusion_optimizer.zero_grad()
        
        instrumental = batch['instrumental'].to(self.device)
        pitch_features = batch['pitch_features'].to(self.device)
        
        # Encode instrumental to latent space
        with torch.no_grad():
            _, _, _, z = self.vae(instrumental)
        
        # Random timesteps
        t = torch.randint(0, self.model_config.diffusion_steps, (z.shape[0],), device=self.device)
        
        # Add noise
        noise = torch.randn_like(z)
        x_t = self.q_sample(z, t, noise)
        
        # Predict noise
        pred_noise = self.diffusion(x_t, t, pitch_features)
        
        # Loss
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.model_config.gradient_clip)
        self.diffusion_optimizer.step()
        
        return {'diffusion_loss': loss.item()}
    
    def validate(self):
        """Validation step"""
        self.vae.eval()
        self.diffusion.eval()
        
        total_vae_loss = 0
        total_diffusion_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # VAE validation
                instrumental = batch['instrumental'].to(self.device)
                recon, mu, logvar, z = self.vae(instrumental)
                
                recon_loss = F.mse_loss(recon, instrumental)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / instrumental.numel()
                vae_loss = recon_loss + 0.001 * kl_loss
                
                # Diffusion validation
                pitch_features = batch['pitch_features'].to(self.device)
                t = torch.randint(0, self.model_config.diffusion_steps, (z.shape[0],), device=self.device)
                noise = torch.randn_like(z)
                x_t = self.q_sample(z, t, noise)
                pred_noise = self.diffusion(x_t, t, pitch_features)
                diffusion_loss = F.mse_loss(pred_noise, noise)
                
                total_vae_loss += vae_loss.item()
                total_diffusion_loss += diffusion_loss.item()
        
        return {
            'val_vae_loss': total_vae_loss / len(self.val_loader),
            'val_diffusion_loss': total_diffusion_loss / len(self.val_loader)
        }
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        
        # Load optimizer states
        self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer'])
        
        # Return epoch to resume from
        return checkpoint['epoch']

    def train(self, resume_from=None):
        """Main training loop with resume capability"""
        wandb.init(project='msldm-voice2instrumental', config=self.model_config)
        
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.model_config.num_epochs):
            # Training
            self.vae.train()
            self.diffusion.train()
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
            
            for batch in pbar:
                # Train VAE
                vae_metrics = self.train_vae_step(batch)
                
                # Train diffusion
                diffusion_metrics = self.train_diffusion_step(batch)
                
                # Update progress bar
                pbar.set_postfix({
                    'VAE': f"{vae_metrics['vae_loss']:.4f}",
                    'Diff': f"{diffusion_metrics['diffusion_loss']:.4f}"
                })
                
                # Log metrics
                wandb.log({**vae_metrics, **diffusion_metrics})
            
            # Validation
            val_metrics = self.validate()
            wandb.log(val_metrics)
            
            # Update schedulers
            self.vae_scheduler.step()
            self.diffusion_scheduler.step()
            
            self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'model_config': self.model_config,
            'data_config': self.data_config
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/msldm_epoch_{epoch}.pt')

def main():
    model_config = ModelConfig()
    data_config = DataConfig(
        vocals_folder='/Users/joaoesteves/mli/SingwithML/03_data_preprocessing/voice',      # ← Update this path
        instruments_folder='/Users/joaoesteves/mli/SingwithML/03_data_preprocessing/instrumental'  # ← Update this path
    )
    
    # Quick fix: Add missing attributes to model_config
    model_config.train_split = data_config.train_split
    model_config.val_split = data_config.val_split
    model_config.test_split = data_config.test_split
    
    # ADD THESE MISSING AUGMENTATION PARAMETERS
    model_config.noise_prob = data_config.noise_prob
    model_config.noise_snr_range = data_config.noise_snr_range
    model_config.pitch_shift_range = data_config.pitch_shift_range
    model_config.time_stretch_range = data_config.time_stretch_range
    model_config.volume_range = data_config.volume_range
    
    trainer = MSLDMTrainer(model_config, data_config)
    trainer.train()

if __name__ == '__main__':
    main()
