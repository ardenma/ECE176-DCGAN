''' Main script '''
import os
import logging
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Generator, Discriminator
from dataloader import get_dataloader
from optim import get_optim

logging.basicConfig(level=logging.DEBUG)
PATH = os.path.dirname(os.path.realpath(__file__))
torch.manual_seed(0)  # For reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# Creating folder for logs and checkpoints
if not os.path.exists(os.path.join(PATH, "logs")):
    os.makedirs(os.path.join(PATH, "logs"))
if not os.path.exists(os.path.join(PATH, "checkpoints")):
    os.makedirs(os.path.join(PATH, "checkpoints"))

g = Generator()
d = Discriminator()
g.to(device=device)
d.to(device=device)

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

# TODO: add argparse and load checkpoint if we need it...
#if load_checkpoint == True:
#    g_checkpoint = torch.load(g_checkpoint_path)
#    d_checkpoint = torch.load(d_checkpoint_path)
#    g.load_state_dict(g_checkpoint['model_state_dict'])
#    d.load_state_dict(d_checkpoint['model_state_dict'])
#    g_optim.load_state_dict(g_checkpoint['optimizer_state_dict'])
#    d_optim.load_state_dict(d_checkpoint['optimizer_state_dict'])
#    epoch = g_checkpoint['epoch']
#    model.eval()
#     - or -
#    model.train()

train_dl = get_dataloader("train", batch_size=128, shape=(64,64), num_workers=6)
#val_dl = get_dataloader("valid", batch_size=128, shape=(64,64), num_workers=6)
#test_dl = get_dataloader("test", batch_size=128, shape=(64,64), num_workers=6)


version = 1       # for checkpointing
num_epochs = 100
k = 2             # number of steps to apply the discriminator, from paper
hparams = {
    "version": version,
    "num_epochs": num_epochs,
    "k": k
}

LOG_DIR = f'version_{version}' + datetime.now().strftime('_%H_%M_%d_%m_%Y')
writer = SummaryWriter(log_dir=os.path.join(PATH, "logs", LOG_DIR))

# Using adversarial training example from https://arxiv.org/pdf/1511.06434.pdf
for epoch in range(1, num_epochs+1):
    print(f"\nBeginning Epoch {epoch}/{num_epochs}.")
    t = tqdm(train_dl)
    for batch in t:
        # Discriminator Training Loop
        logging.debug("Training discriminator.")
        for _ in range(k):
            d_optim.zero_grad()

            # x is batch size x 3 x 218 x 178 by default, 64 x 64 at the end after resize
            # y is batch size x 40
            x = batch[0].to(device)
            batch_size = x.shape[0]
            x_generated = g(torch.rand((batch_size, 100, 1, 1), device=device)) #Generating samples, shape changed from (128, 100)

            # Computing loss for discriminator and optimizing wrt generator
            loss_total = torch.log(d(x))
            loss_g = torch.log(1 - d(x_generated))
            loss_total = torch.sum(torch.cat((loss_total, loss_g))) / batch_size  
            loss_total_neg = -loss_total  # negative sign for gradient ascent instead of descent
            loss_total_neg.backward()
            d_optim.step()
            logging.debug(f"Loss: {loss_total.item()}")

        # Generator Training
        logging.debug("Training generator.")
        g_optim.zero_grad()
        x_generated = g(torch.rand((batch_size, 100, 1, 1), device=device))  # Generating samples, shape changed from (128, 100)

        # Computing loss for generator and optimizing wrt generator
        loss_g = torch.log(d(x))
        loss_g = torch.sum(torch.log(1 - d(x_generated))) / batch_size
        loss_g.backward()
        g_optim.step()
        logging.debug(f"Loss_g: {loss_g.item()}")

        #t.set_postfix({"loss_g": loss_g.item(), "loss_total": loss_total.item()})
        t.set_postfix({"loss": loss_total.item()})
    
    # Logging losses
    #writer.add_scalar("loss", loss_g.item(), epoch)
    writer.add_scalar("loss_total", loss_total.item(), epoch)

    # Checkpointing after every 10 epochs
    if epoch % 10 == 0:
        logging.debug(f"Checkpointing models at epoch {epoch}.")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': g.state_dict(),
                    'optimizer_state_dict': g_optim.state_dict(),
                    'loss': loss_total,
                    'hparams': hparams,
                    }, os.path.join(PATH, f"checkpoints/g_epoch_{epoch}_ver_{version}.pt"))
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': d.state_dict(),
                    'optimizer_state_dict': d_optim.state_dict(),
                    'loss': loss_total,
                    'hparams': hparams,
                    }, os.path.join(PATH, f"checkpoints/d_epoch_{epoch}_ver_{version}.pt"))

# Testing generator output
#x = torch.rand(100)
#g = Generator()
#d = Discriminator()
#y = g(x)
#print(d(y).shape)

