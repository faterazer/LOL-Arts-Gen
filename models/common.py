from collections import OrderedDict
from typing import Tuple

import lightning.pytorch as pl
import torch
from torch import Tensor, nn


class ZLU(pl.LightningModule):
    def __init__(self, min_val: int, max_val: int) -> None:
        super().__init__()
        self.register_buffer("min", torch.FloatTensor([min_val]))
        self.register_buffer("max", torch.FloatTensor([max_val]))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.max(self.min, x)
        x = torch.min(self.max, x)
        return x


act_fn_table = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "silu": nn.SiLU}


def conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Tuple[int, int],
    padding: int = 0,
    stride: int | Tuple[int, int] = 1,
    act_fn: str = "relu",
) -> nn.Module:
    buff = [("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride))]
    if act_fn != "none":
        buff.append(("act_fn", act_fn_table[act_fn]()))
    return nn.Sequential(OrderedDict(buff))


def deconv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Tuple[int, int],
    padding: int = 0,
    stride: int | Tuple[int, int] = 1,
    output_padding: int | Tuple[int, int] = 0,
    act_fn: str = "relu",
) -> nn.Module:
    buff = [
        (
            "deconv",
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                output_padding=output_padding,
            ),
        )
    ]
    if act_fn != "none":
        if act_fn == "zlu":
            buff.append(("act_fn", ZLU(min_val=0, max_val=1)))
        else:
            buff.append(("act_fn", act_fn_table[act_fn]()))
    return nn.Sequential(OrderedDict(buff))
