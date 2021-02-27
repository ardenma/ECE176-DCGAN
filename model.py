''' DCGAN Modeling Code '''
import torch
import torch.nn as nn
import logging
# log = logging.getLogger()
# log.setLevel("DEBUG")
logging.basicConfig(level=logging.DEBUG)


class DCGAN(nn.Module):
    def __init__(self, learning_rate, epochs):
        super(DCGAN, self).__init__()
        self.batch_size = 128
        self.image_dim = 64
        self.channels = 3
        self.lr = learning_rate
        self.n_epochs = epochs

    def forward(self, x):
        pass


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
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.bn1(x)
        logging.debug(x.size())
        x = self.lrelu(self.conv2(x))
        x = self.bn2(x)
        logging.debug(x.size())
        x = self.lrelu(self.conv3(x))
        x = self.bn3(x)
        logging.debug(x.size())
        x = self.lrelu(self.conv4(x))
        logging.debug(x.size())
        return x

# TODO: decide on initialization methods
def weights_init(m):
    if isinstance(m.layer, nn.ConvTranspose2d):
        nn.init.normal_(m.weight)
    if isinstance(m.layer, nn.Conv2d):
        nn.init.normal_(m.weight)
    if isinstance(m.layer, nn.BatchNorm2d):
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)
