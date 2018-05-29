"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
import os
from torch.utils.data.dataset import random_split

def get_mnist(dataset_root, batch_size, train):
    """Get MNIST datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])

    # datasets and data loader
    mnist_dataset = datasets.MNIST(root=os.path.join(dataset_root,'mnist'),
                                   train=train,
                                   transform=pre_process,
                                   download=False)

    # sample 2000 in MNIST
    if train:
        mnist_dataset, mnist_dataset_5k =  random_split(mnist_dataset,[2000, mnist_dataset.__len__()-2000])

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    return mnist_data_loader