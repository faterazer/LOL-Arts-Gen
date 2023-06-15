from argparse import ArgumentParser, Namespace

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from models.autoencoder import AutoEncoder
from datautils import LOLArtsDataset


# Hyperparameters
batch_size = 16
learning_rate = 1e-3
max_epochs = 50


def main(hparams: Namespace) -> None:
    if hparams.accelerator is None:
        hparams.accelerator = "cpu"
    if hparams.devices is None:
        hparams.devices = "auto"

    train_dataset = LOLArtsDataset("./LOL-Arts")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    model = AutoEncoder(learning_rate=learning_rate)
    print(model)
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator=hparams.accelerator, devices=hparams.devices, log_every_n_steps=10
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
