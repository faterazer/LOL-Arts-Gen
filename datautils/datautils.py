import os
from typing import Callable

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class AdaResize(nn.Module):
    def __init__(self, ratio: float) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == 1080:
            return img
        ratio = h / w
        if ratio > self.ratio:
            return transforms.Resize(size=int(1920 * (h / w + 0.1)))(img)
        else:
            return transforms.Resize(size=1080)(img)


default_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        AdaResize(1080 / 1920),
        transforms.RandomCrop(size=(1080, 1920)),
        transforms.ToTensor(),
    ]
)


class LOLArtsDataset(Dataset):
    def __init__(self, data_dir: str, transform: Callable = None) -> None:
        super().__init__()
        self._files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".jpg")]
        self._transform = transform if transform is not None else default_transform

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> Image.Image:
        filename = self._files[index]
        img = Image.open(filename)
        img = self._transform(img)
        return img
