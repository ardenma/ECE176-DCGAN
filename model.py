''' DCGAN Modeling Code '''
import torch
import torch.nn as nn
import logging

class DCGAN(nn.Module):
    def __init__(self, learning_rate, epochs):
        super(DCGAN, self).__init__()
        self.batch_size = 128
        self.image_dim = 64
        self.channels = 3
        self.lr = learning_rate
        self.g = Generator()
        self.d = Discriminator()
        self.apply(weights_init)

    def forward(self, x):
        return self.g(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = nn.Linear(100, 16384)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = torch.reshape(x, (-1, 1024, 4, 4))  # N x C x H x W
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.bn4(x)
        x = self.tanh(self.conv4(x))
        return x


# TODO: need to figure out the architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, 1)
        self.apply(weights_init)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.bn1(x)
        #logging.debug(x.size())
        x = self.lrelu(self.conv2(x))
        x = self.bn2(x)
        #logging.debug(x.size())
        x = self.lrelu(self.conv3(x))
        x = self.bn3(x)
        #logging.debug(x.size())
        x = self.lrelu(self.conv4(x))
        #logging.debug(x.size())
        x = self.fc(self.flatten(x))
        return x

# TODO: decide on initialization methods
# DCGAN paper initializes using zero-centered Gaussian distribution, with standard deviation of 0.2

def weights_init(m):
    if hasattr(m, "weight"):
        logging.debug(f"Initializing weight for {m} weights")
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
    if hasattr(m, "bias"):
        if m.bias is not None:
            logging.debug(f"Initializing weight for {m} bias")
            nn.init.normal_(m.bias, mean=0.0, std=0.2)

def weights_init_old(m):
    if isinstance(m.layer, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
    if isinstance(m.layer, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
    if isinstance(m.layer, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        nn.init.normal_(m.bias, mean=0.0, std=0.2)
        #nn.init.zeros_(m.bias)
    if isinstance(m.layer, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        nn.init.zeros_(m.bias, mean=0.0, std=0.2)
