import cupy as cp
import numpy as np
import time

def benchmark_transfer(size_in_bytes, direction, use_pinned_memory=False, iterations=10):
    """
    Benchmarks data transfer between CPU and GPU.

    Args:
        size_in_bytes (int): The size of the data to transfer in bytes.
        direction (str): 'htod' (Host to Device) or 'dtoh' (Device to Host).
        use_pinned_memory (bool): Whether to use page-locked (pinned) memory.
        iterations (int): Number of times to run the transfer for averaging.
    """
    if use_pinned_memory:
        # cupy.cuda.alloc_pinned_memory returns a memory pointer
        # We need to wrap it in a numpy array to use it easily.
        host_mem_ptr = cp.cuda.alloc_pinned_memory(size_in_bytes)
        host_array = np.frombuffer(host_mem_ptr, dtype=np.uint8, count=size_in_bytes)
    else:
        # Standard pageable numpy array
        host_array = np.random.randint(0, 255, size=size_in_bytes, dtype=np.uint8)

    device_array = cp.empty(size_in_bytes, dtype=cp.uint8)

    # Use CUDA events for accurate timing
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()

    # --- Warm-up transfer ---
    # This ensures the GPU is awake and any one-time setup costs are paid.
    if direction == 'htod':
        device_array.set(host_array)
    else: # dtoh
        _ = device_array.get()
    cp.cuda.Stream.null.synchronize()


    # --- Timed transfers ---
    start_event.record()
    for _ in range(iterations):
        if direction == 'htod':
            # Host (CPU) to Device (GPU)
            device_array.set(host_array)
        else: # dtoh
            # Device (GPU) to Host (CPU)
            _ = device_array.get()
    stop_event.record()
    stop_event.synchronize()

    # Calculate elapsed time and bandwidth
    total_time_ms = cp.cuda.get_elapsed_time(start_event, stop_event)
    avg_time_s = (total_time_ms / 1000) / iterations
    bandwidth_gbps = (size_in_bytes / (1024**3)) / avg_time_s
    
    return bandwidth_gbps

if __name__ == "__main__":
    # Test with various data sizes (64MB, 128MB, 256MB, 512MB, 1GB)
    sizes_mb = [64, 128, 256, 512, 1024]
    sizes_bytes = [s * 1024 * 1024 for s in sizes_mb]

    print("-" * 70)
    print(f"{'Data Size':<12} | {'Direction':<12} | {'Memory Type':<15} | {'Bandwidth (GB/s)':<20}")
    print("-" * 70)

    for size_b, size_mb in zip(sizes_bytes, sizes_mb):
        # 1. Standard (Pageable) Memory
        htod_speed = benchmark_transfer(size_b, 'htod', use_pinned_memory=False)
        dtoh_speed = benchmark_transfer(size_b, 'dtoh', use_pinned_memory=False)

        print(f"{size_mb:<10}MB | {'CPU -> GPU':<12} | {'Pageable':<15} | {htod_speed:<20.2f}")
        print(f"{size_mb:<10}MB | {'GPU -> CPU':<12} | {'Pageable':<15} | {dtoh_speed:<20.2f}")

        # 2. Pinned Memory
        htod_pinned_speed = benchmark_transfer(size_b, 'htod', use_pinned_memory=True)
        dtoh_pinned_speed = benchmark_transfer(size_b, 'dtoh', use_pinned_memory=True)

        print(f"{size_mb:<10}MB | {'CPU -> GPU':<12} | {'Pinned':<15} | {htod_pinned_speed:<20.2f}")
        print(f"{size_mb:<10}MB | {'GPU -> CPU':<12} | {'Pinned':<15} | {dtoh_pinned_speed:<20.2f}")
        print("-" * 70)
