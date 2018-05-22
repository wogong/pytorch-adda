"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

import params


def get_svhn(train):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Grayscale(),
                                      transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    if train == True:
        svhn_dataset = datasets.SVHN(root=params.data_root,
                                   split='train',
                                   transform=pre_process,
                                   download=True)
    else:
        svhn_dataset = datasets.SVHN(root=params.data_root,
                                   split='test',
                                   transform=pre_process,
                                   download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return svhn_data_loader
