''' Main script '''
import torch
from model import Generator

# Testing generator output
x = torch.rand(100)
model = Generator()
model(x)