from argparse import ArgumentParser, Namespace

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from AE import AutoEncoder
from datautils import LOLArtsDataset


def main(hparams: Namespace) -> None:
    if hparams.accelerator is None:
        hparams.accelerator = "cpu"
    if hparams.devices is None:
        hparams.devices = "auto"

    train_dataset = LOLArtsDataset("./LOL-Arts")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    model = AutoEncoder()
    trainer = pl.Trainer(max_epochs=5, accelerator=hparams.accelerator, devices=hparams.devices)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
