''' DCGAN Modeling Code '''
import torch
import torch.nn as nn
import logging 
#log = logging.getLogger()
#log.setLevel("DEBUG")
logging.basicConfig(level=logging.DEBUG)

class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
    
    def forward(self, x):
        pass

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = nn.Linear(100, 16384)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # TODO: H x W dimensions seem slightly off... results in (_, 3, 85, 85), expected (_, 3, 64, 64)
    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = torch.reshape(x, (-1, 1024, 4,4))  # N x C x H x W
        x = self.bn1(x)
        logging.debug(x.size())
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        logging.debug(x.size())
        x = self.relu(self.conv2(x))
        x = self.bn3(x)
        logging.debug(x.size())
        x = self.relu(self.conv3(x))
        x = self.bn4(x)
        logging.debug(x.size())
        x = self.tanh(self.conv4(x))
        logging.debug(x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
    def forward(self, x):
        pass