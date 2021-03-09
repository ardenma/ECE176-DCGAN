''' Dataloader and utilities for loading data'''
import logging

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_dataloader(split, batch_size, shape, num_workers):
    logging.info(f"Getting {split} dataloader.")
    
    # download the data set and apply transformation with cropping and normalization
    '''
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.CenterCrop(shape),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    '''
    transform = transforms.Compose([transforms.Resize(shape),
                                    transforms.ToTensor()])
    
    celeba_data = dset.CelebA('.', split=split, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(celeba_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=torch.cuda.is_available())
    
    return data_loader