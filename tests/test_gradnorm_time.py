import torch
import time
from torch.func import grad

# --- 1. Optimization Flags ---
# Enable TensorFloat32 for ~8x speedup on Ampere+ GPUs (RTX 30xx/40xx, A100)
torch.set_float32_matmul_precision('high')

# --- 2. Setup Data ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N, D = 1000, 500
lambda_reg = 0.1
lr = 0.01

print(f"Device: {device.upper()}")
print(f"PyTorch: {torch.__version__}")
print("-" * 50)

# Create Data
X = torch.randn(N, D, device=device)
y = torch.randn(N, 1, device=device)

# --- 3. Define Functions ---

# Approach A: Standard Autograd
def standard_step(w_std, optimizer):
    optimizer.zero_grad()
    pred = X @ w_std
    mse_loss = (pred - y).pow(2).sum()
    
    # create_graph=True is needed for higher-order derivatives
    grad_mse = torch.autograd.grad(mse_loss, w_std, create_graph=True)[0]
    grad_l1_norm = torch.abs(grad_mse).sum()
    
    total_loss = mse_loss + (lambda_reg * grad_l1_norm)
    total_loss.backward()
    optimizer.step()

# Approach B: torch.func + Compile
def compute_mse(params, x_data, y_data):
    pred = x_data @ params
    return ((pred - y_data) ** 2).sum()

def total_objective(params, x_data, y_data):
    mse = compute_mse(params, x_data, y_data)
    # Functional gradient inside objective
    mse_grads = grad(compute_mse)(params, x_data, y_data)
    grad_l1_reg = torch.abs(mse_grads).sum()
    return mse + (lambda_reg * grad_l1_reg)

@torch.compile
def compiled_step_fn(w, x, y):
    # Functional gradient of total objective
    grads = grad(total_objective)(w, x, y)
    w.sub_(lr * grads)
    return w

# --- 4. Manual Benchmark Loop ---

def benchmark_cuda(name, func, args, iterations=100):
    # Warmup
    print(f"Running {name}...", end="", flush=True)
    for _ in range(10):
        func(*args)
    torch.cuda.synchronize()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (elapsed_time_ms / iterations) * 1000
    print(f" Done! {avg_time_us:.1f} us")
    return avg_time_us

# --- Run Benchmarks ---

# 1. Standard Autograd
w_std = torch.randn(D, 1, device=device, requires_grad=True)
opt_std = torch.optim.SGD([w_std], lr=lr)
time_std = benchmark_cuda("Standard Autograd", standard_step, (w_std, opt_std))

# 2. Compiled torch.func
w_func = w_std.detach().clone()
# We must compile first (implicit in first call)
try:
    print("Compiling...", end="", flush=True)
    compiled_step_fn(w_func, X, y) # Triggers compilation
    print(" Success.")
    
    time_func = benchmark_cuda("torch.func + Compile", compiled_step_fn, (w_func, X, y))

    # --- Print Comparison ---
    print("\n" + "=" * 30)
    print(f"Standard Autograd: {time_std:.1f} us")
    print(f"Compiled Func:     {time_func:.1f} us")
    print(f"Speedup:           {time_std / time_func:.2f}x")
    print("=" * 30)

except Exception as e:
    print(f"\nCompilation Failed: {e}")