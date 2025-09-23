import torch


print("PyTorch version:", torch.__version__)

if not torch.cuda.is_available():
    print("CUDA is not available on this system.")
else:
    print("CUDA Version (used by PyTorch):", torch.version.cuda)
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA-enabled device(s).")
    
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        device = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(device)
        print(f"  Name: {props.name}")
        print(f"  CUDA Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Streaming Multiprocessor (SM) Count: {props.multi_processor_count}")

