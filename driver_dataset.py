from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class DriverDataset(Dataset):
    def __init__(self, frame, image_root, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.class_to_idx = {f"c{i}": i for i in range(10)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.image_root / row["classname"] / row["img"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.class_to_idx[row["classname"]]
        return image, label