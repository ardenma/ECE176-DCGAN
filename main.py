''' Main script '''
import logging

import torch
from tqdm import tqdm

from model import Generator, Discriminator
from dataloader import get_dataloader
from optim import get_optim

logging.basicConfig(level=logging.DEBUG)

train_dl = get_dataloader("train", batch_size=128, shape=(64,64), num_workers=6)
val_dl = get_dataloader("valid", batch_size=128, shape=(64,64), num_workers=6)
test_dl = get_dataloader("test", batch_size=128, shape=(64,64), num_workers=6)

g = Generator()
d = Discriminator()

g_optim_options = {
    "lr": 0.0002, 
    "eps": 1e-08,
    "weight_decay": 0, 
    "amsgrad": False
}

d_optim_options = {
    "lr": 0.0002,
    "eps": 1e-08,
    "weight_decay": 0, 
    "amsgrad": False
}

g_optim = get_optim("Adam", parameters=g.parameters(), options=g_optim_options)
d_optim = get_optim("Adam", parameters=d.parameters(), options=d_optim_options)

num_epochs = 100
k = 1  # number of steps to apply the discriminator, from paper

# Using adversarial training example from https://arxiv.org/pdf/1511.06434.pdf
for epoch in range(num_epochs):
    print(f"\nBeginning Epoch {epoch}/{num_epochs}.\n")
    for batch in tqdm(train_dl):
        # Discriminator Training Loop
        logging.debug("Training discriminator.")
        for _ in range(k):
            d_optim.zero_grad()

            # x is batch size x 3 x 218 x 178 by default, 64 x 64 at the end after resize
            # y is batch size x 40
            x, y = batch
            batch_size = x.shape[0]
            x_generated = g(torch.rand((batch_size, 100)))  # Generating samples

            # Computing loss for discriminator and optimizing wrt generator
            loss_data = torch.log(d(x))
            loss_g = torch.log(1 - d(x_generated))
            loss_total = -torch.sum(torch.cat((loss_data, loss_g))) / batch_size  # negative sign for gradient ascent instead of descent
            loss_total.backward()
            d_optim.step()

        # Generator Training
        logging.debug("Training generator.")
        g_optim.zero_grad()
        x_generated = g(torch.rand((batch_size, 100)))  # Generating samples

        # Computing loss for generator and optimizing wrt generator
        loss_data = torch.log(d(x))
        loss_g = torch.sum(torch.log(1 - d(x_generated))) / batch_size
        loss_g.backward()
        g_optim.step()


# Testing generator output
#x = torch.rand(100)
#g = Generator()
#d = Discriminator()
#y = g(x)
#print(d(y).shape)
