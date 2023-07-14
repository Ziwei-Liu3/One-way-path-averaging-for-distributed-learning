import torch
import torch.distributed as dist
import os 
import datetime 

output_dir = "./output.tmp"

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

try:
    dist_backend = getattr(args, 'dist_backend', 'nccl')
    torch.distributed.init_process_group(backend=dist_backend)
    print("NCCL backend is available.")
except torch.distributed.BackendUnavailable:
    print("NCCL backend is not available.")

try:
    from torch.distributed import nccl
    print("NCCL backend is available.")
except ImportError:
    print("NCCL backend is not available.")

try:
    from torch.distributed import gloo
    print("Gloo backend is available.")
except ImportError:
    print("Gloo backend is not available.")