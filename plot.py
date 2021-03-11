import os
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(PATH, 'figures')):
    os.makedirs(os.path.join(PATH, 'figures'))

def plot_loss(G_losses:list, D_losses:list, title:str, filename:str) -> None:
    assert isinstance(G_losses, list) and isinstance(D_losses, list)
    assert isinstance(title, str) and isinstance(filename, str)
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH, 'figures', filename))
    plt.show()
    
def plot_generated(imgs:list, version:int) -> None:
    assert isinstance(imgs, list)
    assert isinstance(version, int)
    for i, img in enumerate(imgs):
        plt.figure(figsize=(15,15))
        plt.axis('off')
        plt.title(f'Generated Images Epoch{i+1} Version{version}')
        plt.imshow(np.transpose(img, (1,2,0)))
        plt.savefig(os.path.join(PATH, 'figures', f'generated_img_ver{version}_epoch{i+1}.jpg'))
        plt.show()