import torchrdit_rs
import torch

from torchrdit_rs.wrap import wrap_torch_tensor_add_one, wrap_torch_tensor_multiply


if __name__ == "__main__":
    print(torchrdit_rs.sum_as_string(20, 22))

    print(torchrdit_rs.hello())

    print(torchrdit_rs.fibonacci_number_map([1, 2, 3, 4, 5, 6, 7]))

    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], requires_grad=True, device="mps")
    print(f"input_tensor: {input_tensor}")
    output_tensor = wrap_torch_tensor_add_one(input_tensor)
    print(f"output_tensor: {output_tensor}")
    sum_output_tensor = torch.sum(output_tensor)
    print(sum_output_tensor)
    sum_output_tensor.backward()
    print(input_tensor.grad)
    print(output_tensor.device)