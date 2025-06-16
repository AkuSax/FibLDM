import torch
from metrics import RealismMetrics

# Helper to print metric results and types
def print_metric_result(name, result):
    print(f"  {name}: {result} (type: {type(result)})")
    if isinstance(result, tuple):
        for i, val in enumerate(result):
            print(f"    tuple[{i}]: {val} (type: {type(val)})")


def test_metrics():
    """Test the metrics with various inputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = RealismMetrics(device=device)
    
    # Test with realistic batch sizes and channels
    batch_sizes = [1, 2, 10, 50, 100]
    channels = [1]  # We only use 1-channel images
    
    for batch_size in batch_sizes:
        for num_channels in channels:
            print(f"\nTesting batch_size={batch_size}, channels={num_channels}")
            
            # Create more realistic test data
            real_images = torch.rand(batch_size, num_channels, 256, 256, device=device)  # Using rand instead of randn for more realistic values
            fake_images = torch.rand(batch_size, num_channels, 256, 256, device=device)
            same_mask_images = torch.rand(batch_size, num_channels, 256, 256, device=device)
            
            # Test individual metrics
            try:
                # Convert to 3 channels for FID
                real_3ch = real_images.repeat(1, 3, 1, 1)
                fake_3ch = fake_images.repeat(1, 3, 1, 1)
                fid = metrics.fid(real_3ch, fake_3ch)
                print(f"- FID: {fid} (type: {type(fid)})")
            except Exception as e:
                print(f"- FID: FID ERROR: {str(e)}")
                
            try:
                # Convert to 3 channels for KID
                real_3ch = real_images.repeat(1, 3, 1, 1)
                fake_3ch = fake_images.repeat(1, 3, 1, 1)
                kid = metrics.kid(real_3ch, fake_3ch)
                print(f"- KID: {kid} (type: {type(kid)})")
                if isinstance(kid, tuple):
                    print(f"  tuple[0]: {kid[0]} (type: {type(kid[0])})")
                    print(f"  tuple[1]: {kid[1]} (type: {type(kid[1])})")
            except Exception as e:
                print(f"- KID: KID ERROR: {str(e)}")
                
            try:
                # Convert to 3 channels and normalize to [-1,1] for LPIPS
                real_lpips = (real_images * 2 - 1).repeat(1, 3, 1, 1)
                fake_lpips = (fake_images * 2 - 1).repeat(1, 3, 1, 1)
                same_mask_lpips = (same_mask_images * 2 - 1).repeat(1, 3, 1, 1)
                lpips = metrics.lpips(fake_lpips, same_mask_lpips)
                print(f"- LPIPS: {lpips} (type: {type(lpips)})")
            except Exception as e:
                print(f"- LPIPS: LPIPS ERROR: {str(e)}")
                
            try:
                ssim = metrics.ssim(fake_images, real_images)
                print(f"- SSIM: {ssim} (type: {type(ssim)})")
            except Exception as e:
                print(f"- SSIM: SSIM ERROR: {str(e)}")
            
            # Test compute_metrics function
            print("- All metrics via compute_metrics:")
            all_metrics = metrics.compute_metrics(real_images, fake_images, same_mask_images)
            print(f"  compute_metrics: {all_metrics}")
    
    # Test edge cases
    print("\nTesting edge cases:")
    
    # Case 1: Empty batch
    print("\nCase 1: Empty batch")
    empty_real = torch.empty(0, 1, 256, 256, device=device)
    empty_fake = torch.empty(0, 1, 256, 256, device=device)
    empty_same_mask = torch.empty(0, 1, 256, 256, device=device)
    empty_result = metrics.compute_metrics(empty_real, empty_fake, empty_same_mask)
    print(f"  Empty batch result: {empty_result}")
    
    # Case 2: Invalid values (NaN, inf)
    print("\nCase 2: Invalid values")
    invalid_real = torch.full((2, 1, 256, 256), float('nan'), device=device)
    invalid_fake = torch.full((2, 1, 256, 256), float('inf'), device=device)
    invalid_same_mask = torch.full((2, 1, 256, 256), float('nan'), device=device)
    invalid_result = metrics.compute_metrics(invalid_real, invalid_fake, invalid_same_mask)
    print(f"  Invalid values result: {invalid_result}")
    
    # Case 3: Different batch sizes
    print("\nCase 3: Different batch sizes")
    diff_real = torch.rand(5, 1, 256, 256, device=device)
    diff_fake = torch.rand(3, 1, 256, 256, device=device)
    diff_same_mask = torch.rand(5, 1, 256, 256, device=device)
    diff_result = metrics.compute_metrics(diff_real, diff_fake, diff_same_mask)
    print(f"  Different batch sizes result: {diff_result}")
    
    # Case 4: Realistic fibrosis-like patterns
    print("\nCase 4: Realistic patterns")
    # Create some basic patterns that might resemble fibrosis
    pattern_real = torch.zeros(2, 1, 256, 256, device=device)
    pattern_fake = torch.zeros(2, 1, 256, 256, device=device)
    pattern_same_mask = torch.zeros(2, 1, 256, 256, device=device)
    
    # Add some basic patterns
    for i in range(2):
        # Add some random lines and curves
        for _ in range(10):
            # Generate random start and end points
            start_x = torch.randint(0, 256, (1,), device=device).item()
            start_y = torch.randint(0, 256, (1,), device=device).item()
            end_x = torch.randint(0, 256, (1,), device=device).item()
            end_y = torch.randint(0, 256, (1,), device=device).item()
            
            # Draw a line
            x = torch.linspace(start_x, end_x, 100, device=device)
            y = torch.linspace(start_y, end_y, 100, device=device)
            x = x.long().clamp(0, 255)
            y = y.long().clamp(0, 255)
            pattern_real[i, 0, y, x] = 1.0
            
            # Draw a slightly different line for fake
            x = x + torch.randint(-5, 6, x.shape, device=device)
            y = y + torch.randint(-5, 6, y.shape, device=device)
            x = x.long().clamp(0, 255)
            y = y.long().clamp(0, 255)
            pattern_fake[i, 0, y, x] = 1.0
            
            # Same mask has same pattern as real
            pattern_same_mask[i, 0, y, x] = 1.0
    
    pattern_result = metrics.compute_metrics(pattern_real, pattern_fake, pattern_same_mask)
    print(f"  Realistic patterns result: {pattern_result}")

if __name__ == "__main__":
    test_metrics()