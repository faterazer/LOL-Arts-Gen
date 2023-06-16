from collections import OrderedDict
from typing import Tuple

import lightning.pytorch as pl
import math
import torch
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import LambdaLR


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


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    :param optimizer: The optimizer for which to schedule the learning rate.
    :param num_warmup_steps: The number of steps for the warmup phase.
    :param num_training_steps: The total number of training steps.
    :param num_cycles: The number of waves in the cosine schedule (the defaults is to just decrease from the max value
    to 0 following a half-cosine).
    :param last_epoch: The index of the last epoch when resuming training.
    :return: `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
