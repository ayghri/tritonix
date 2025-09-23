from typing import Callable, List, Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn
import torch


def cat_keep_shapes(
    x_list: List[Tensor],
) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(
    flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]
) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [
        shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes
    ]
    outputs_reshaped = [
        o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)
    ]
    return outputs_reshaped


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(
            in_features, hidden_features, bias=bias, device=device
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias, device=device
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(
            in_features, swiglu_hidden_features, bias=bias, device=device
        )
        self.w2 = nn.Linear(
            in_features, swiglu_hidden_features, bias=bias, device=device
        )
        self.w3 = nn.Linear(
            swiglu_hidden_features, out_features, bias=bias, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(
            in_features, 2 * swiglu_hidden_features, bias=bias, device=device
        )

        self.w3 = nn.Linear(
            swiglu_hidden_features, out_features, bias=bias, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        # hidden = F.silu(x1) * x2
        hidden = torch.nn.functional.glu(x1, dim=-1)
        return self.w3(hidden)


if __name__ == "__main__":
    D = 1024
    H = 1024 * 4
    layer1 = (
        SwiGLUFFN(
            in_features=D,
            hidden_features=H,
            out_features=D,
            bias=False,
            align_to=64,
        )
        .to("cuda")
        .half()
    )
    layer2 = (
        SwiGLUFFNFused(
            in_features=D,
            hidden_features=H,
            out_features=D,
            bias=False,
            align_to=64,
        )
        .to("cuda")
        .half()
    )

    # inp = torch.randn(2048, D, device="cuda", dtype=torch.float16)

    import triton
    import triton.testing

    @triton.testing.perf_report(
        [
            triton.testing.Benchmark(
                x_names=[
                    "N"
                ],  # argument names to use as an x-axis for the plot
                x_vals=[
                    2**i for i in range(8, 13)
                ],  # different values of `N` to take in the experiment
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "layer1",
                    "layer2",
                ],  # values for `line_arg`
                line_names=[
                    "SwiGLUFFN",
                    "SwiGLUFFNFused",
                ],  # legend names for the lines
                styles=[("blue", "-"), ("green", "-")],  # line styles
                ylabel="GB/s",  # label name for the y-axis
                plot_name="swiglu-performance",  # name for the plot. Used also as a file name for saving the plot.
                # args={"M": 4096, "dtype": torch.float16},
                args={"dtype": torch.float16},
            )
        ]
    )
    def benchmark(N, provider, dtype):
        x = torch.randn(N, D, device="cuda", dtype=dtype)
        if provider == "layer1":
            l_fn = layer1
        else:
            l_fn = layer2

        # layer = torch.compile(lambda x: l_fn(x), fullgraph=True)
        layer = l_fn
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer(x), quantiles=quantiles
        )

        def gbps(ms):
            # return 2 * x.nelement() * x.element_size() / ms * 1e-6
            return (
                2
                * (
                    2 * N * l_fn.w1.in_features * l_fn.w3.in_features
                    + N * l_fn.w3.in_features * l_fn.w3.out_features
                )
                / 1024**4
                / (ms * 1e-3)
            )

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(show_plots=True, print_data=True)
