import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_step=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device='cuda', schedule_name='linear'):
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size
        self.schedule_name = schedule_name

        self.beta = self.prepare_noise_schedule().to(device) # beta from t=0 to t=T
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.is_latent = False
    
    def prepare_noise_schedule(self):
        if self.schedule_name == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_step)
        elif self.schedule_name == 'cosine':
            # Cosine schedule implementation
            steps = self.noise_step + 1
            x = torch.linspace(0, self.noise_step, steps)
            alphas_cumprod = torch.cos(((x / self.noise_step) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_name}")

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        return (x_t - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat

    def sample_timesteps(self, n): # sample n timesteps from the noise schedule
        return torch.randint(low=1, high = self.noise_step, size=(n,))

    def sample(self, model, n, condition=None, fast_sampling=False, latent_dim=None):
        # Debug: Diffusion.sample - Initial call parameters
        print(f"Diffusion.sample called - n: {n}, is_latent: {getattr(self, 'is_latent', False)}, img_size: {self.img_size}, latent_dim: {latent_dim}, condition provided: {condition is not None}")
        model.eval()
        with torch.no_grad():
            is_latent = getattr(self, 'is_latent', False)
            if is_latent:
                if latent_dim is None:
                    raise ValueError("latent_dim must be specified for latent space sampling.")
                x = torch.randn(n, latent_dim, self.img_size, self.img_size, device=self.device)
                # Debug: Initial random noise tensor for sampling
                print(f"Diffusion.sample - Initial noisy x shape: {x.shape}")
            else:
                x = torch.randn(n, 1, self.img_size, self.img_size, device=self.device)
            
            # Process in smaller chunks to save memory
            chunk_size = 16 if fast_sampling else 4  # Larger chunks for validation
            total_chunks = (n - 1) // chunk_size + 1
            
            # Use fewer steps for validation
            steps = 100 if fast_sampling else self.noise_step
            step_indices = torch.linspace(0, self.noise_step - 1, steps).long()
            
            for i in range(0, n, chunk_size):
                end_idx = min(i + chunk_size, n)
                x_chunk = x[i:end_idx]
                condition_chunk = condition[i:end_idx] if condition is not None else None
                
                # Only show progress bar for non-fast sampling and when not too many chunks
                show_progress = not fast_sampling and total_chunks <= 4
                if show_progress:
                    tqdm_bar = tqdm(
                        reversed(step_indices),
                        desc=f"Denoising chunk {i//chunk_size + 1}/{total_chunks}",
                        leave=False
                    )
                    iterator = tqdm_bar
                else:
                    tqdm_bar = None
                    iterator = reversed(step_indices)
                
                for t in iterator:
                    t_batch = torch.ones(end_idx - i).to(self.device) * t
                    
                    # For FiLM-based models, do not concatenate; always use x_in = x_chunk
                    x_in = x_chunk
                    
                    # Debug: Denoising loop - x_in shape and t_batch value
                    print(f"  Denoising t={t.item()}: x_in shape: {x_in.shape}, t_batch: {t_batch[0].item()}")
                    if condition_chunk is not None:
                        print(f"  Denoising t={t.item()}: condition_chunk shape: {condition_chunk.shape}")
                    
                    # Predict noise
                    if is_latent and condition_chunk is not None:
                        predicted_noise = model(x_in, t_batch, condition_chunk)
                    else:
                        predicted_noise = model(x_in, t_batch)
                    
                    # Denoise step
                    alpha = self.alpha[t]
                    alpha_hat = self.alpha_hat[t]
                    beta = self.beta[t]
                    
                    if t > 0:
                        noise = torch.randn_like(x_chunk)
                    else:
                        noise = torch.zeros_like(x_chunk)
                        
                    x_chunk = 1 / torch.sqrt(alpha) * (x_chunk - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    
                    # Update progress bar with current noise level (only if showing progress)
                    if tqdm_bar is not None:
                        tqdm_bar.set_postfix({'noise_level': f'{t/self.noise_step:.2%}'})
                
                x[i:end_idx] = x_chunk
                
        model.train()
        
        # For latent space, don't rescale - keep in original range
        if is_latent:
            # Debug: Final generated sample properties
            print(f"Diffusion.sample - Final generated x shape: {x.shape}, Min: {x.min():.4f}, Max: {x.max():.4f}")
            return x
        else:
            # Rescale from [-1, 1] to [0, 1] for image space
            return (x.clamp(-1, 1) + 1) / 2
    