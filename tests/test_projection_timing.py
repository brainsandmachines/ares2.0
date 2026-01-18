import torch
import time
from ares.utils.adv import L1Step, L2Step, LinfStep

# Timing function
def time_projection(projection_fn, x, orig_input, eps, iterations=100):
    start_time = time.time()
    for _ in range(iterations):
        projection_fn.project(x)
    end_time = time.time()
    return (end_time - start_time) * 1000 / iterations  # Time in ms

# Test the timing differences
def main():
    batch_size = 32
    channels = 3
    height = 224
    width = 224
    eps = 0.1

    x = torch.rand(batch_size, channels, height, width)
    orig_input = torch.rand(batch_size, channels, height, width)

    l2_step = L2Step(orig_input, eps, step_size=0.01)
    linf_step = LinfStep(orig_input, eps, step_size=0.01)
    l1_step = L1Step(orig_input, eps, step_size=0.01)

    l2_time = time_projection(l2_step, x, orig_input, eps)
    linf_time = time_projection(linf_step, x, orig_input, eps)
    l1_time = time_projection(l1_step, x, orig_input, eps)

    print(f"L2 Projection Time: {l2_time:.2f} ms")
    print(f"Linf Projection Time: {linf_time:.2f} ms")
    print(f"L1 Projection Time: {l1_time:.2f} ms")

if __name__ == "__main__":
    main()