import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datautils import LOLArtsDataset, test_transform
from models.autoencoder import AutoEncoder

# Hyperparameters
batch_size = 32
ckpt_path = "./lightning_logs/MK-4/C/checkpoints/epoch=79-val_l2_loss=2253.61.ckpt"

test_dataset = LOLArtsDataset("./OOD", transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

model = AutoEncoder.load_from_checkpoint(ckpt_path).eval()
l1_losses, l2_losses = [], [] 

for x in tqdm(test_dataloader):
    with torch.no_grad():
        x = x.cuda()
        outputs = model(x)
    l1_loss = F.l1_loss(outputs, x, reduction="none").sum(dim=[1, 2, 3])
    l1_losses.extend(l1_loss.tolist())
    l2_loss = F.mse_loss(outputs, x, reduction="none").sum(dim=[1, 2, 3])
    l2_losses.extend(l2_loss.tolist())

mean_l1_loss = sum(l1_losses) / len(l1_losses)
mean_l2_loss = sum(l2_losses) / len(l2_losses)
print("平均 L1 损失:", mean_l1_loss)
print("平均 L2 损失:", mean_l2_loss)
