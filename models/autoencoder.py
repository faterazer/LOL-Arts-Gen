import lightning.pytorch as pl
import torch.optim
from lightning.pytorch.utilities import grad_norm
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from models.common import conv_layer, deconv_layer, get_cosine_schedule_with_warmup


class Encoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_layer(3, 3, kernel_size=5, padding=2, act_fn="gelu"),
            conv_layer(3, 3, kernel_size=5, padding=2, act_fn="gelu"),
            conv_layer(3, 32, kernel_size=5, padding=2, stride=(3, 4), act_fn="gelu"),
        )  # (32, 360, 480)
        self.stage2 = nn.Sequential(
            conv_layer(32, 256, kernel_size=5, padding=2, stride=(3, 4), act_fn="gelu"),
        )  # (256, 120, 120)
        self.stage3 = nn.Sequential(
            conv_layer(256, 512, kernel_size=5, padding=2, stride=3, act_fn="gelu")
        )  # (512, 40, 40)
        self.stage4 = nn.Sequential(
            conv_layer(512, 1024, kernel_size=3, padding=1, stride=2, act_fn="tanh"),
        )  # (1024, 20, 20)

        self.res1 = nn.Conv2d(32, 512, kernel_size=(9, 12), stride=(9, 12))

    def forward(self, x: Tensor) -> Tensor:
        y = self.stage1(x)
        y, r = self.stage2(y), self.res1(y)
        y = self.stage3(y) + r
        y = self.stage4(y)
        return y.flatten()


class Decoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.stage4 = nn.Sequential(
            deconv_layer(1024, 512, kernel_size=3, padding=1, stride=2, output_padding=1, act_fn="gelu")
        )  # (512, 40, 40)
        self.stage3 = nn.Sequential(
            deconv_layer(512, 256, kernel_size=5, padding=2, stride=3, output_padding=2, act_fn="gelu")
        )  # (256, 120, 120)
        self.stage2 = nn.Sequential(
            deconv_layer(256, 32, kernel_size=5, padding=2, stride=(3, 4), output_padding=(2, 3), act_fn="gelu")
        )  # (32, 360, 480)
        self.stage1 = nn.Sequential(
            deconv_layer(32, 3, kernel_size=5, padding=2, stride=(3, 4), output_padding=(2, 3), act_fn="zlu")
        )  # (3, 1080, 1920)

    def forward(self, x: Tensor) -> None:
        y = x.reshape(-1, 1024, 20, 20)
        y = self.stage4(y)
        y = self.stage3(y)
        y = self.stage2(y)
        y = self.stage1(y)
        return y


class AutoEncoder(pl.LightningModule):
    def __init__(self, max_iters: int, use_lr_scheduler: bool = False, learning_rate: float = 1e-3):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # Freeze
        # self.encoder.stage1.requires_grad_(False)
        # self.encoder.stage2.requires_grad_(False)
        # self.encoder.stage3.requires_grad_(False)
        # self.encoder.stage4.requires_grad_(False)
        # self.encoder.res1.requires_grad_(False)
        # self.decoder.stage4.requires_grad_(False)
        # self.decoder.stage3.requires_grad_(False)
        # self.decoder.stage2.requires_grad_(False)
        # self.decoder.stage1.requires_grad_(False)

        self.lr = learning_rate
        self.max_iters = max_iters
        self.use_lr_scheduler = use_lr_scheduler
        self.save_hyperparameters()

        self.validation_step_outputs = []

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

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

    def validation_step(self, batch: Tensor, *args) -> None:
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        l1_loss = F.l1_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3])
        l2_loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3])
        self.validation_step_outputs.append((l1_loss.tolist(), l2_loss.tolist()))

    def on_validation_epoch_end(self) -> None:
        l1_loss_list = []
        l2_loss_list = []
        for l1_loss, l2_loss in self.validation_step_outputs:
            l1_loss_list.extend(l1_loss)
            l2_loss_list.extend(l2_loss)
        mean_l1_loss = sum(l1_loss_list) / len(l1_loss_list)
        mean_l2_loss = sum(l2_loss_list) / len(l2_loss_list)
        self.log("val_l1_loss", mean_l1_loss)
        self.log("val_l2_loss", mean_l2_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> optim.Optimizer | dict:
        # params = list(self.named_parameters())

        # def is_pretrained(name: str) -> bool:
        #     return "conv_a" in name or "deconv_a" in name

        # grouped_parameters = [
        #     {"params": [p for n, p in params if is_pretrained(n)], "lr": self.lr / 100},
        #     {"params": [p for n, p in params if not is_pretrained(n)], "lr": self.lr},
        # ]
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        if self.use_lr_scheduler:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * self.max_iters), num_training_steps=self.max_iters
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}
        else:
            return optimizer

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norm = grad_norm(self, norm_type=2)
        self.log_dict(norm)
