import torch
import time

def benchmark_torch_transfer(size_in_bytes, direction, use_pinned_memory=False, iterations=10):
    """
    Benchmarks data transfer between CPU and GPU using PyTorch.

    Args:
        size_in_bytes (int): The size of the data to transfer in bytes.
        direction (str): 'htod' (Host to Device) or 'dtoh' (Device to Host).
        use_pinned_memory (bool): Whether to use page-locked (pinned) memory.
        iterations (int): Number of times to run the transfer for averaging.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")

    device = torch.device('cuda')
    
    # Create the source tensor on the CPU
    # The .pin_memory() method makes it a page-locked tensor.
    cpu_tensor = torch.randint(0, 255, (size_in_bytes,), dtype=torch.uint8, device='cpu')
    if use_pinned_memory:
        cpu_tensor = cpu_tensor.pin_memory()
    
    # Create the source tensor on the GPU
    gpu_tensor = torch.randint(0, 255, (size_in_bytes,), dtype=torch.uint8, device=device)

    # Use CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- Warm-up transfer ---
    # This ensures any lazy initializations are done before we start timing.
    if direction == 'htod':
        _ = cpu_tensor.to(device, non_blocking=True)
    else: # dtoh
        _ = gpu_tensor.to('cpu', non_blocking=True)
    torch.cuda.synchronize() # Wait for the warm-up to complete


    # --- Timed transfers ---
    start_event.record()
    for _ in range(iterations):
        if direction == 'htod':
            # Host (CPU) to Device (GPU)
            _ = cpu_tensor.to(device, non_blocking=True)
        else: # dtoh
            # Device (GPU) to Host (CPU)
            # We copy to the same pinned tensor to ensure fair measurement
            _ = gpu_tensor.to(cpu_tensor.device, non_blocking=True)
    end_event.record()

    # Wait for all queued operations to complete
    torch.cuda.synchronize()

    # Calculate elapsed time and bandwidth
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_s = (total_time_ms / 1000) / iterations
    bandwidth_gbps = (size_in_bytes / (1024**3)) / avg_time_s
    
    return bandwidth_gbps

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("PyTorch CUDA is not available. Aborting benchmark.")
        exit()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test with various data sizes (64MB, 128MB, 256MB, 512MB, 1GB)
    sizes_mb = [64, 128, 256, 512, 1024]
    sizes_bytes = [s * 1024 * 1024 for s in sizes_mb]

    print("-" * 70)
    print(f"{'Data Size':<12} | {'Direction':<12} | {'Memory Type':<15} | {'Bandwidth (GB/s)':<20}")
    print("-" * 70)

    for size_b, size_mb in zip(sizes_bytes, sizes_mb):
        # 1. Standard (Pageable) Memory
        htod_speed = benchmark_torch_transfer(size_b, 'htod', use_pinned_memory=False)
        dtoh_speed = benchmark_torch_transfer(size_b, 'dtoh', use_pinned_memory=False)

        print(f"{size_mb:<10}MB | {'CPU -> GPU':<12} | {'Pageable':<15} | {htod_speed:<20.2f}")
        print(f"{size_mb:<10}MB | {'GPU -> CPU':<12} | {'Pageable':<15} | {dtoh_speed:<20.2f}")

        # 2. Pinned Memory
        htod_pinned_speed = benchmark_torch_transfer(size_b, 'htod', use_pinned_memory=True)
        dtoh_pinned_speed = benchmark_torch_transfer(size_b, 'dtoh', use_pinned_memory=True)

        print(f"{size_mb:<10}MB | {'CPU -> GPU':<12} | {'Pinned':<15} | {htod_pinned_speed:<20.2f}")
        print(f"{size_mb:<10}MB | {'GPU -> CPU':<12} | {'Pinned':<15} | {dtoh_pinned_speed:<20.2f}")
        print("-" * 70)
