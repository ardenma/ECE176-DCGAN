''' Main script '''
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import plot
from model import Generator, Discriminator, weights_init
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
    
# Initialize Generator and Discriminator
g = Generator().to(device=device)         # Default generator depth is 128
d = Discriminator().to(device=device)     # Default discriminator depth is 128
g.apply(weights_init)
d.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCEWithLogitsLoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Test with tutorial parameters
g_optim_options = {
    "lr": 0.0002, 
    "betas": (0.5, 0.999)
}

d_optim_options = {
    "lr": 0.0002,
    "betas": (0.5, 0.9999)
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


version = 10       # for checkpointing
num_epochs = 20
K = 1             # number of steps to apply the discriminator, from paper
hparams = {
    "version": version,
    "num_epochs": num_epochs,
    "k": K
}

LOG_DIR = f'version_{version}' + datetime.now().strftime('_%H_%M_%d_%m_%Y')
writer = SummaryWriter(log_dir=os.path.join(PATH, "logs", LOG_DIR))

# Store training losses
g_losses = []
d_losses = []

# Store generated images
img_list = []

# Using adversarial training example from https://arxiv.org/pdf/1511.06434.pdf
for epoch in range(1, num_epochs+1):
    print(f"\nBeginning Epoch {epoch}/{num_epochs}.")
    t = tqdm(train_dl)
    for batch in t:
        #========== Discriminator Training Loop ==========#
        logging.debug("Training discriminator.")
        for k in range(K):
            # Real batch
            d.zero_grad()
            x = batch[0].to(device)
            batch_size = x.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = d(x).view(-1)
            loss_real = criterion(output, label)
            loss_real.backward()
            D_x = output.mean().item()

            # Fake batch
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = g(noise)
            label.fill_(fake_label)
            output = d(fake.detach()).view(-1)
            loss_fake = criterion(output, label)
            loss_fake.backward()
            D_G_z1 = output.mean().item()
            
            loss_total = loss_real + loss_fake
            d_optim.step()
            
            logging.debug(f"Loss_d_{k}: {loss_total.item()}")
        #=================================================#

        
        #========== Generator Training ==========#
        
        logging.debug("Training generator.")
        g.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = d(fake).view(-1)
        # Calculate G's loss based on this output
        loss_g = criterion(output, label)
        # Calculate gradients for G
        loss_g.backward()
        D_G_z2 = output.mean().item()
        # Update G
        g_optim.step()
        
        logging.debug(f"Loss_g: {loss_g.item()}")

        #t.set_postfix({"loss_g": loss_g.item(), "loss_total": loss_total.item()})
        t.set_postfix({"loss": loss_total.item()})
        #========================================#
        
        # store losses
        g_losses.append(loss_g.item())
        d_losses.append(loss_total.item())
    
    #writer.add_scalar("loss", loss_g.item(), epoch)
    writer.add_scalar("loss_total", loss_total.item(), epoch)

    # Saving generated images after every epoch
    with torch.no_grad():
        fake = g(torch.randn(64, 100, 1, 1, device=device)).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # Checkpointing after every epoch
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

plot.plot_loss(g_losses, d_losses, f'Generator and Discriminator Training Loss Version {version}', f'train_loss_ver_{version}.png')
plot.plot_generated(img_list, version)

