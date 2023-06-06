import os
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class LOLArtsDataset(Dataset):
    def __init__(self, data_dir: str, transform: Callable = None) -> None:
        super().__init__()
        self._files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".jpg")]
        self._transform = transform

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> Image.Image:
        filename = self._files[index]
        img = Image.open(filename)
        if self._transform:
            img = self._transform(img)
        return img
