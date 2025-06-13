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

    def sample(self, model, n, contour): 
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 1, self.img_size, self.img_size).to(self.device) # gray scale image
            
            for i in tqdm(reversed(range(self.noise_step)),position=0): 
                t = (torch.ones(n) * i).long().to(self.device)
                xt = torch.cat((x, contour), dim=1) # Concatenate contour to the image
                predicted_noise = model(xt, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha) * (x - ((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise) + torch.sqrt(beta)*noise
        model.train()
        x = (x.clamp(-1,1)+1)/2
        return x
    