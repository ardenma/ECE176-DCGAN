''' Main script '''
import torch
from model import Generator, Discriminator

# Testing generator output
x = torch.rand(100)
g = Generator()
d = Discriminator()
y = g(x)
print(d(y).shape)

