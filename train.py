from argparse import ArgumentParser, Namespace

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from datautils import LOLArtsDataset, test_transform
from models.autoencoder import AutoEncoder

# Hyperparameters
batch_size = 16
learning_rate = 1e-3
max_epochs = 100


def main(hparams: Namespace) -> None:
    if hparams.accelerator is None:
        hparams.accelerator = "cpu"
    if hparams.devices is None:
        hparams.devices = "auto"

    train_dataset = LOLArtsDataset("./LOL-Arts")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_dataset = LOLArtsDataset("./LOL-Arts", transform=test_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, num_workers=8, shuffle=False, pin_memory=True)

    # model = AutoEncoder.load_from_checkpoint(
    #     "./lightning_logs/version_40/checkpoints/epoch=99-val_l2_loss=8320.56.ckpt",
    #     strict=True,
    #     max_iters=max_epochs * len(train_loader),
    #     use_lr_scheduler=True,
    #     learning_rate=learning_rate,
    # )
    model = AutoEncoder(max_iters=max_epochs * len(train_loader), use_lr_scheduler=False, learning_rate=learning_rate)
    print(model)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(
        monitor="val_l2_loss", filename="{epoch}-{val_l2_loss:.2f}", save_top_k=3, every_n_epochs=5
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        callbacks=[lr_monitor, ckpt_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
