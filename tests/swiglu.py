from typing import Callable, Optional
from xformers.ops import SwiGLU

from torch import Tensor, nn
import torch.nn.functional as F
import torch
import triton


class GLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        # x1, x2 = x12.chunk(2, dim=-1)
        # hidden = F.silu(x1) * x2
        hidden = F.glu(x12, dim=-1)
        return self.w3(hidden)
        # return hidden
        return x12


class GLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        # print(f"Using hidden_features={hidden_features} for SwiGLUFFNFused")
        # hidden_features2 = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        # print(f"hidden_features2={hidden_features2}")

        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


B = 1024 * 4
D = 1024 * 2
K = 1024 * 8
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

x = torch.randn(B, D, dtype=DTYPE, device=DEVICE)


torch_glu = GLUFFN(
    in_features=D, hidden_features=K, out_features=D, bias=False
).to(DEVICE, dtype=DTYPE)
xform_glu = GLUFFNFused(
    in_features=D, hidden_features=K, out_features=D, bias=False
).to(DEVICE, dtype=DTYPE)

for p in torch_glu.parameters():
    p.requires_grad_(False)
for p in xform_glu.parameters():
    p.requires_grad_(False)


# @torch.no_grad()
# def torch_swiglu(x: Tensor) -> Tensor:
def torch_swiglu() -> Tensor:
    return torch_glu(x)


# @torch.no_grad()
# def xformers_swiglu(x: Tensor) -> Tensor:
def xformers_swiglu() -> Tensor:
    return xform_glu(x)


# compiled_torch_swiglu = torch.compile(torch_swiglu, fullgraph=True)
# compiled_xformers_swiglu = torch.compile(xformers_swiglu, fullgraph=True)
compiled_torch_swiglu = torch.compile(torch_swiglu)
compiled_xformers_swiglu = torch.compile(xformers_swiglu)

print(
    "Benchmarking SwiGLU implementations for batch size",
    B,
    "and input dimension",
    D,
    "hidden size",
    K,
)
for name, fn in [
    ("torch_swiglu", torch_swiglu),
    ("compiled_torch_swiglu", compiled_torch_swiglu),
    ("compiled_xformers_swiglu", compiled_xformers_swiglu),
    ("xformers_swiglu", xformers_swiglu),
]:
    print(f"Running {name}...")
    torch.cuda.synchronize()
    ms, min_ms, max_ms = triton.testing.do_bench(
        fn,
        quantiles=[0.5, 0.2, 0.8],
        warmup=200,
        rep=200,
    )  # type: ignore[no-untyped-call]
    print(
        f"{name} took {ms:.2f} ms (min: {min_ms:.2f} ms, max: {max_ms:.2f} ms)"
    )
