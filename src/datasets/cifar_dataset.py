import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

class test_cifar_dataset(Dataset):
    """
    A custom dataset class for CIFAR-10 test data with balanced binary classification.
    
    Args:
        data_len (int): Number of samples to include in the dataset
        prob_dist (list): Probability distribution for class sampling [p_airplane, p_not_airplane]
    """
    def __init__(self, data_len, prob_dist):
        self.data_len = data_len
        self.prob_dist = prob_dist
        
        # Load CIFAR-10 test dataset
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
        ])
        
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        # Filter for airplane (class 0) and non-airplane images
        self.airplane_indices = [i for i, (_, label) in enumerate(self.dataset) if label == 0]
        self.non_airplane_indices = [i for i, (_, label) in enumerate(self.dataset) if label != 0]
        
        # Sample indices based on probability distribution
        self.sampled_indices = self._sample_indices()
        
    def _sample_indices(self):
        """Sample indices based on the specified probability distribution."""
        sampled_indices = []
        for _ in range(self.data_len):
            # Sample class based on probability distribution
            class_idx = np.random.choice(2, p=self.prob_dist)
            
            if class_idx == 0:  # Airplane
                idx = np.random.choice(self.airplane_indices)
            else:  # Not airplane
                idx = np.random.choice(self.non_airplane_indices)
                
            sampled_indices.append(idx)
            
        return sampled_indices
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        # Get the actual index from our sampled indices
        actual_idx = self.sampled_indices[idx]
        image, label = self.dataset[actual_idx]
        
        # Convert to binary label (0 for airplane, 1 for not-airplane)
        binary_label = 0 if label == 0 else 1
        
        return image, binary_label 