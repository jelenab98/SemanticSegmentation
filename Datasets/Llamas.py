from pathlib import Path

from torch.utils.data import Dataset


class BinaryLlamasDataset(Dataset):
    """LLAMAS dataset for lane tracking."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 2

    def __init__(self,
                 root: Path,
                 transforms: lambda x: x,
                 subset: str = "train",
                 color: str = "grayscale"
                 ) -> None:

        if color == "color":
            self.num_classes = 5

        self.transforms = transforms
        self.subset = subset
        self.root = root
        self.images, self.labels = [], []

        folder_names = [line.rstrip() for line in (self.root / f"{self.subset}_folders.txt").open("r").readlines()]

        for folder in folder_names:
            self.images.extend((self.root / f"{color}_images" / self.subset / folder).glob("*.png"))
            self.labels.extend((self.root / "labels" / self.subset / folder).glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        data = {
            "image": self.images[item],
            "name": self.images[item].stem,
            "subset": self.subset,
            "labels": self.labels[item]
        }
        return self.transforms(data)
