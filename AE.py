import lightning.pytorch as pl
import torch.optim
from torch import Tensor, nn, optim
from torch.nn import functional as F


class AutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=15, padding=7),  # (4, 1080, 1920)
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=15, padding=7, stride=(3, 4)),  # (8, 360, 480)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=15, padding=7, stride=(3, 4)),  # (16, 120, 120)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=7, padding=3, stride=3),  # (32, 40, 40)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=2),  # (64, 20, 20)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),  # (128, 10, 10)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # (256, 5, 5)
            nn.ReLU(),
            nn.Conv2d(256, 40, kernel_size=1),  # (40, 5, 5)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 256, kernel_size=1),  # (256, 5, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1),  # (128, 10, 10)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, stride=2, output_padding=1),  # (64, 20, 20)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, padding=3, stride=2, output_padding=1),  # (32, 40, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=7, padding=3, stride=3, output_padding=2),  # (16, 120, 120),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=15, padding=7, stride=(3, 4), output_padding=(2, 3)),  # (8, 360, 480)
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 4, kernel_size=15, padding=7, stride=(3, 4), output_padding=(2, 3)
            ),  # (4, 1080, 1920)
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=15, padding=7),  # (3, 1080, 1920)
            nn.ReLU(),
        )
        self._lr = learning_rate

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return optimizer
