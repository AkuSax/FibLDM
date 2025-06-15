import torch
from metrics import RealismMetrics

# Helper to print metric results and types
def print_metric_result(name, result):
    print(f"  {name}: {result} (type: {type(result)})")
    if isinstance(result, tuple):
        for i, val in enumerate(result):
            print(f"    tuple[{i}]: {val} (type: {type(val)})")


def test_metrics():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = RealismMetrics(device=device)
    
    # Test various batch sizes and channel counts
    for batch_size in [1, 2, 10, 50, 100]:
        for channels in [1, 3]:
            print(f"\nTesting batch_size={batch_size}, channels={channels}")
            real = torch.rand(batch_size, channels, 256, 256, device=device)
            fake = torch.rand(batch_size, channels, 256, 256, device=device)
            try:
                # Test FID only
                print("- FID:")
                try:
                    metrics.fid.reset()
                    metrics.fid.update(real.repeat(1,3,1,1) if channels==1 else real, real=True)
                    metrics.fid.update(fake.repeat(1,3,1,1) if channels==1 else fake, real=False)
                    fid_result = metrics.fid.compute()
                    print_metric_result("FID", fid_result)
                except Exception as e:
                    print(f"  FID ERROR: {e}")
                # Test KID only
                print("- KID:")
                try:
                    from torchmetrics.image.kid import KernelInceptionDistance
                    min_subset_size = min(real.shape[0], fake.shape[0], 50)
                    kid_metric = KernelInceptionDistance(subset_size=min_subset_size, normalize=True).to(device)
                    kid_metric.update(real.repeat(1,3,1,1) if channels==1 else real, real=True)
                    kid_metric.update(fake.repeat(1,3,1,1) if channels==1 else fake, real=False)
                    kid_result = kid_metric.compute()
                    print_metric_result("KID", kid_result)
                except Exception as e:
                    print(f"  KID ERROR: {e}")
                # Test LPIPS only
                print("- LPIPS:")
                try:
                    lpips_result = metrics.lpips(
                        fake.repeat(1,3,1,1) if channels==1 else fake,
                        real.repeat(1,3,1,1) if channels==1 else real
                    )
                    print_metric_result("LPIPS", lpips_result)
                except Exception as e:
                    print(f"  LPIPS ERROR: {e}")
                # Test SSIM only
                print("- SSIM:")
                try:
                    ssim_result = metrics.ssim(fake, real)
                    print_metric_result("SSIM", ssim_result)
                except Exception as e:
                    print(f"  SSIM ERROR: {e}")
                # Test all together via compute_metrics
                print("- All metrics via compute_metrics:")
                try:
                    result = metrics.compute_metrics(real, fake, same_mask_images=real)
                    print(f"  compute_metrics: {result}")
                except Exception as e:
                    print(f"  compute_metrics ERROR: {e}")
            except Exception as e:
                print(f"  GENERAL ERROR: {e}")

    # Test edge case: batch smaller than KID subset size
    print("\nTesting edge case: batch_size=3, channels=1 (KID subset_size > batch)")
    real = torch.rand(3, 1, 256, 256, device=device)
    fake = torch.rand(3, 1, 256, 256, device=device)
    try:
        result = metrics.compute_metrics(real, fake, same_mask_images=real)
        print(f"  Success: {result}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_metrics() 