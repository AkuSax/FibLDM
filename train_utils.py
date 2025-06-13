import torch
import torch.nn as nn
import math
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler

class EMA:
    """
    Maintains exponential moving average of model parameters.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class EarlyStopper:
    """
    Early stopping based on validation IoU.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.FloatTensor:
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def evaluate_iou(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    ious = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            t = torch.zeros(images.size(0), dtype=torch.long, device=device)
            x_in = torch.cat((images, masks), dim=1)
            logits = model(x_in, t)
            preds = (logits > 0.5).float()
            inter = (preds * masks).sum(dim=[1,2,3])
            union = ((preds + masks) >= 1).float().sum(dim=[1,2,3])
            ious.append((inter / (union + 1e-6)).mean().item()) 
    return sum(ious) / len(ious)



def train(
    model: torch.nn.Module,
    diffusion,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    ema_decay: float = 0.9999,
    early_stop_patience: int = 10,
    sample_batch_size: int = 16,
    save_interval: int = 30,
    use_amp: bool = True,
    use_compile: bool = True
):
    torch.backends.cudnn.benchmark = True
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, backend="inductor")
    
    
    ema = EMA(model, decay=ema_decay)
    stopper = EarlyStopper(patience=early_stop_patience)
    scaler = GradScaler("cuda")
    loss_fn = nn.MSELoss()

    model.to(device)
    best_iou = 0.0

    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0

        for images, contour in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # move to GPU with non_blocking
            images = images.to(device,   non_blocking=True)
            contour = contour.to(device,  non_blocking=True)
            # sample noise
            t = diffusion.sample_timesteps(images.size(0)).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            # prep network input
            x_in = torch.cat((x_t, contour), dim=1)
            # AMP‐wrapped forward/backward
            optimizer.zero_grad()
            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    pred_noise = model(x_in, t)
                    loss = loss_fn(pred_noise, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_noise = model(x_in, t)
                loss = loss_fn(pred_noise, noise)
                loss.backward()
                optimizer.step()

            # EMA update & bookkeeping
            ema.update(model)
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} | Train MSE: {avg_loss:.6f}")

        # Validation IoU (every epoch)
        current_iou = evaluate_iou(model, val_loader, device)
        print(f"Epoch {epoch} | Val IoU: {current_iou:.4f}  (best: {best_iou:.4f})")

        # Save best (EMA shadow) model
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save({
                "model_state":     model.state_dict(),
                "ema_state":       ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "epoch":           epoch,
                "best_iou":        best_iou
            }, "best_ddpm_model.pth")
            print("Saved new best model.")

        # Early stopping
        if stopper.step(current_iou):
            print("Early stopping criterion met; halting training.")
            break

        # Periodic checkpoints
        if epoch % save_interval == 0:
            ckpt_path = os.path.join("trained_models", f"ddpmv2_model_{epoch:03d}.pt")
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch":           epoch,
                "mse":             avg_loss,
                "iou":             current_iou
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final sampling with EMA weights
    print("Loading EMA weights for sampling…")
    ema.apply_shadow(model)
    model.eval()

    # Get a batch of contours
    sample_contour = next(iter(val_loader))[1][:sample_batch_size].to(device)
    samples = diffusion.sample(model, sample_batch_size, sample_contour)

    ema.restore(model)
    return samples
