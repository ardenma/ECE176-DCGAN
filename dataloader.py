''' Dataloader and utilities for loading data'''
import logging

import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize

def get_dataloader(split, batch_size, shape, num_workers):
    logging.info(f"Getting {split} dataloader.")
    transform = Compose([Resize(shape), ToTensor()])  # TODO: add other transforms/data augmentation
    celeba_data = torchvision.datasets.CelebA('.', split=split, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(celeba_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=torch.cuda.is_available())
    return data_loader