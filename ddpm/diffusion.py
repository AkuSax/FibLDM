import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_step=1000, beta_start=1e-4, beta_end=0.02, img_size = 256, device='cuda'):
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size

        self.beta = self.prepare_noise_schedule().to(device) # beta from t=0 to t=T
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_step)

    def noise_image(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def sample_timesteps(self, n): # sample n timesteps from the noise schedule
        return torch.randint(low=1, high = self.noise_step, size=(n,))

    def sample(self, model, n, condition=None, fast_sampling=False):
        model.eval()
        with torch.no_grad():
            # Get the underlying model if wrapped in DDP
            if hasattr(model, 'module'):
                model = model.module
            
            # Initialize with pure noise
            x = torch.randn(n, 1, self.img_size, self.img_size).to(self.device)
            
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
                
                # Create progress bar for this chunk
                pbar = tqdm(
                    reversed(step_indices),
                    desc=f"Denoising chunk {i//chunk_size + 1}/{total_chunks}",
                    leave=False
                )
                
                for t in pbar:
                    t_batch = torch.ones(end_idx - i).to(self.device) * t
                    
                    # Prepare input with condition
                    if condition_chunk is not None:
                        x_in = torch.cat([x_chunk, condition_chunk], dim=1)
                    else:
                        x_in = x_chunk
                    
                    # Predict noise
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
                    
                    # Update progress bar with current noise level
                    pbar.set_postfix({'noise_level': f'{t/self.noise_step:.2%}'})
                
                x[i:end_idx] = x_chunk
                
        model.train()
        return x
    