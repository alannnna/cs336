import torch
from torch import Tensor
from jaxtyping import Float


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.w: Float[Tensor, "d_model"] = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Upcast to float32 for our calculation, then cast back at the end
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rm = (x ** 2 + self.eps).sum(-1) / self.d_model
        rms = rm.sqrt()
        result = (x / rms) * self.w

        return result.to(in_dtype)

