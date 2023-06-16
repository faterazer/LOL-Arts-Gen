from collections import OrderedDict

import lightning.pytorch as pl
import torch.optim
from torch import Tensor, nn, optim
from torch.nn import functional as F

from models.common import conv_layer, deconv_layer, get_cosine_schedule_with_warmup


class AutoEncoder(pl.LightningModule):
    def __init__(self, warmup: float, max_iters: int, learning_rate: float = 1e-3):
        super().__init__()
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("conv_a1", conv_layer(3, 3, kernel_size=5, padding=2, act_fn="relu")),
                    ("conv_a2", conv_layer(3, 3, kernel_size=5, padding=2, act_fn="relu")),
                    (
                        "conv_a3",
                        conv_layer(3, 32, kernel_size=5, padding=2, stride=(3, 4), act_fn="tanh"),
                    ),  # (32, 360, 480)
                ]
            )
        )
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    # deconv_layer(2048, 1024, kernel_size=3, padding=1, stride=2, output_padding=1),  # (1024, 10, 10)
                    # deconv_layer(1024, 512, kernel_size=5, padding=2, stride=2, output_padding=1),  # (512, 20, 20)
                    # deconv_layer(512, 256, kernel_size=7, padding=3, stride=2, output_padding=1),  # (256, 40, 40)
                    # deconv_layer(256, 128, kernel_size=7, padding=3, stride=3, output_padding=2),  # (128, 120, 120)
                    # deconv_layer(256, 32, kernel_size=15, padding=7, stride=(3, 4), output_padding=(2, 3)),  # (32, 360, 480)
                    (
                        "deconv_a",
                        deconv_layer(
                            32, 3, kernel_size=5, padding=2, stride=(3, 4), output_padding=(2, 3), act_fn="zlu"
                        ),
                    )  # (3, 1080, 1920)
                ]
            )
        )
        self.warmup = warmup
        self.max_iters = max_iters
        self.lr = learning_rate
        self.save_hyperparameters()

    @staticmethod
    def _reconstruction_loss(x_hat: Tensor, x: Tensor) -> Tensor:
        loss = F.mse_loss(x_hat, x, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=0)
        return loss

    def training_step(self, batch: Tensor) -> Tensor:
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self._reconstruction_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.warmup * self.max_iters), num_training_steps=self.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
