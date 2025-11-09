import argparse
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def test_build_dataset():
    args = argparse.Namespace(
        train_dir="/storage/test/bml_group/tomerash/datasets/imagenet_sample/train",
        eval_dir="/storage/test/bml_group/tomerash/datasets/imagenet_sample/val",
        batch_size=8,
        workers=0,
        aug_splits=0,
        distributed=False,
    )

    # ✅ add a basic transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.ImageFolder(root=args.train_dir, transform=transform)
    dataset_eval  = datasets.ImageFolder(root=args.eval_dir, transform=transform)

    # ✅ sample smaller subset for quick testing
    small_train = Subset(dataset_train, random.sample(range(len(dataset_train)), 200))
    small_eval  = Subset(dataset_eval, random.sample(range(len(dataset_eval)), 100))

    loader_train = DataLoader(small_train, batch_size=args.batch_size, shuffle=True)
    loader_eval  = DataLoader(small_eval, batch_size=args.batch_size, shuffle=False)

    print("✅ Loaded small sample successfully.")
    print(f"Train batches: {len(loader_train)}")
    print(f"Eval batches:  {len(loader_eval)}")

    # ✅ this now works
    data, target = next(iter(loader_train))
    print("Batch shapes:", data.shape, target.shape)

if __name__ == "__main__":
    test_build_dataset()
