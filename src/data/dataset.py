"""Oxford-IIIT Pet thin wrapper returning (image, trimap, breed_name, species)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image
import torchvision.datasets as dsets


def _species(breed: str) -> str:
    """Infer species: Oxford cats are CamelCase, dogs are snake_case."""
    return "cat" if breed[0].isupper() else "dog"


class OxfordPetDataset:
    """Combined train+val+test split with (image, trimap, breed, species)."""

    def __init__(self, root: str = "data", size: int = 512):
        self.size = size
        trainval = dsets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types=["category", "segmentation"],
            download=False,
        )
        test = dsets.OxfordIIITPet(
            root=root,
            split="test",
            target_types=["category", "segmentation"],
            download=False,
        )
        self._ds = [trainval, test]
        self._classes = trainval.classes
        self._lengths = [len(trainval), len(test)]
        self.breed_names = [
            self._classes[trainval[i][1][0]] for i in range(len(trainval))
        ] + [
            self._classes[test[i][1][0]] for i in range(len(test))
        ]

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image, str, str]:
        if idx < self._lengths[0]:
            ds, local_idx = self._ds[0], idx
        else:
            ds, local_idx = self._ds[1], idx - self._lengths[0]

        image, (cat_idx, trimap) = ds[local_idx]
        breed = self._classes[cat_idx]
        species = _species(breed)

        image = image.convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        trimap = trimap.resize((self.size, self.size), Image.NEAREST)
        return image, trimap, breed, species

    @property
    def classes(self) -> list[str]:
        return self._classes
