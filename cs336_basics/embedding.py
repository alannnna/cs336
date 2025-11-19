import torch
from torch import Tensor
from jaxtyping import Float, Int


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.w: Float[Tensor, "vocab_size d_model"] = \
            torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.w, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        # return torch.index_select(self.w, 0, token_ids)
        return self.w[token_ids]

