import os
import matplotlib.pyplot as plt

def plot_loss(G_losses:list, D_losses:list, title:str, filename:str) -> None:
    assert isinstance(G_losses, list) and isinstance(D_losses, list)
    assert isinstance(title, str) and isinstance(filename, str)
    PATH = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(PATH, 'figures')):
        os.makedirs(os.path.join(PATH, 'figures'))
    
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH, 'figures', filename))
    plt.show()
    
def plot_img(imgs:list, title:str, filename:str) -> None:
    assert isinstance(imgs, list)
    assert isinstance(title, str) and isinstance(filename, str)
    pass