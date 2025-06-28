from torchrdit_rs import torch_tensor_add_one, torch_tensor_multiply
import torch

def wrap_torch_tensor_add_one(tensor: torch.Tensor) -> torch.Tensor:
    return torch_tensor_add_one(tensor)

def wrap_torch_tensor_multiply(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch_tensor_multiply(tensor, other)