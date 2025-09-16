import torch
import torch.nn as nn
import math
from torch import Tensor
import einops as eo
from .transformers.perceiver import PerceiverEncoder


def make_cross_attn_mask(
    source_pad_mask: Tensor,
    target_pad_mask: Tensor,
    num_heads: int,
) -> Tensor:
    target_len = target_pad_mask.size(1)

    attn_mask = eo.repeat(
        tensor=source_pad_mask,
        pattern='n s -> (n h) t s',
        h=num_heads,
        t=target_len,
    )
    return attn_mask


def make_self_attn_mask(
    pad_mask: Tensor,
    num_heads: int,
) -> Tensor:
    return make_cross_attn_mask(
        source_pad_mask=pad_mask,
        target_pad_mask=pad_mask,
        num_heads=num_heads
    )


@torch.no_grad()
def init_params(module: nn.Module) -> None:
    """
    Initialize parameters for various module types with appropriate strategies.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, PerceiverEncoder):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.latent)
        scale = 1
        n = max(1, fan_in)
        s = scale / n
        stddev = math.sqrt(s)
        stddev = stddev / .87962566103423978
        nn.init.trunc_normal_(module.latent, std=stddev, a=-2, b=+2)
