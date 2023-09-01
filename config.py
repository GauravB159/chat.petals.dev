from dataclasses import dataclass
from typing import Optional

import torch

from cpufeature import CPUFeature


@dataclass
class ModelInfo:
    repo: str
    adapter: Optional[str] = None
    name: Optional[str] = None


MODELS = [
    ModelInfo(repo="codellama/CodeLlama-34b-Instruct-hf")
]
DEFAULT_MODEL_NAME = "codellama/CodeLlama-34b-Instruct-hf"

# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
INITIAL_PEERS = ["/ip4/142.58.37.127/tcp/31337/p2p/QmTowvf8FEhPPKPjWZBQt24GnqrtzwTPeLiKZmYgA687nw"]

DEVICE = "cuda"

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
MAX_SESSIONS = 50  # Has effect only for API v1 (HTTP-based)
