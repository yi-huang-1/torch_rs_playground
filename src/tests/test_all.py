import pytest
import torchrdit_rs
import torch

def test_sum_as_string():
    assert torchrdit_rs.sum_as_string(1, 1) == "2"

def test_hello():
    assert torchrdit_rs.hello() == "Hello, World!"

def test_fibonacci():
    assert torchrdit_rs.fibonacci(1) == 1
    assert torchrdit_rs.fibonacci(2) == 1
    assert torchrdit_rs.fibonacci(3) == 2
    assert torchrdit_rs.fibonacci(4) == 3
    assert torchrdit_rs.fibonacci(5) == 5
    assert torchrdit_rs.fibonacci(6) == 8
    assert torchrdit_rs.fibonacci(7) == 13

def test_fibonacci_number_map():
    assert torchrdit_rs.fibonacci_number_map([1, 2, 3, 4, 5, 6, 7]) == {
        "1": 1,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 5,
        "6": 8,
        "7": 13,
    }

def test_torch_tensor_add_one():
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True, device="cpu")
    output_tensor = torchrdit_rs.torch_tensor_add_one(input_tensor)
    assert output_tensor.grad_fn is not None
    assert output_tensor.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def test_torch_tensor_multiply():
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True, device="cpu")
    other_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True, device="cpu")
    output_tensor = torchrdit_rs.torch_tensor_multiply(input_tensor, other_tensor)
    assert output_tensor.grad_fn is not None
    assert output_tensor.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0]