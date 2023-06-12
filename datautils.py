import os
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def AdaResize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if h == 1080:
        return img
    ratio = h / w
    if ratio > 1080 / 1920:
        return transforms.Resize(size=int(1920 * (h / w + 0.1)))(img)
    else:
        return transforms.Resize(size=1080)(img)


default_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        AdaResize,
        transforms.RandomCrop(size=(1080, 1920)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.3104, 0.2722, 0.3082), std=(0.2035, 0.1932, 0.1918)),
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
