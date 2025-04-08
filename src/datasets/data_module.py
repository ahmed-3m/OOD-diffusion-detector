import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import lightning as L

class DiffusionData(L.LightningDataModule):
    def __init__(self, batch_size=32):
        """
        Data module for training unconditional diffusion models on CIFAR-10 airplanes.
        
        Args:
            batch_size (int): Batch size for training and validation dataloaders
        """
        super().__init__()
        self.batch_size = batch_size
        self.augment = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def prepare_data(self):
        """Download the dataset."""
        load_dataset("cifar10")

    def preprocess_train(self, examples):
        """Process dataset examples for training."""
        images = examples["img"]
        examples["images"] = [self.augment(image) for image in images]
        return examples

    def filter_airplanes(self, dataset):
        """Filter dataset to keep only airplane class (class 0)."""
        return dataset.filter(lambda example: example["label"] == 0)

    def train_dataloader(self):
        """Return the training dataloader with only airplane images."""
        dataset = load_dataset("cifar10")
        dataset = dataset["train"]
        # Filter to keep only airplane class
        dataset = self.filter_airplanes(dataset)
        dataset = dataset.with_transform(self.preprocess_train)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        """Return the validation dataloader with only airplane images."""
        dataset = load_dataset("cifar10")
        dataset = dataset["test"]
        # Filter to keep only airplane class
        dataset = self.filter_airplanes(dataset)
        dataset = dataset.with_transform(self.preprocess_train)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def collate_fn(self, batch):
        """Collate a batch of images."""
        images = torch.stack([item["images"] for item in batch])  # Stack tensors into a batch
        return {"images": images} 