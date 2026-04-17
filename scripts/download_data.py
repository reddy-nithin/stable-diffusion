"""Download Oxford-IIIT Pet dataset with trimaps."""
import pathlib
import torchvision.datasets as dsets


def main():
    root = pathlib.Path("data")
    root.mkdir(exist_ok=True)
    print("Downloading Oxford-IIIT Pet dataset (images + trimaps)...")
    dsets.OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types=["category", "segmentation"],
        download=True,
    )
    dsets.OxfordIIITPet(
        root=str(root),
        split="test",
        target_types=["category", "segmentation"],
        download=True,
    )
    print(f"Done. Data saved to {root.resolve()}/oxford-iiit-pet/")


if __name__ == "__main__":
    main()
