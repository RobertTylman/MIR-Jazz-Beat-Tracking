import torch
import time

print("Starting dummy GPU keep-alive...", flush=True)
try:
    while True:
        # Perform matrix multiplications to keep the GPU busy
        x = torch.randn(2048, 2048, device="cuda")
        y = torch.matmul(x, x)
        time.sleep(1.0)
except Exception as e:
    print(f"Dummy script stopped: {e}")
