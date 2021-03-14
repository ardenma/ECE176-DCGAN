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
    def __init__(self, depth=128):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(100, depth*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(depth*8)
        
        self.conv2 = nn.ConvTranspose2d(depth*8, depth*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth*4)

        self.conv3 = nn.ConvTranspose2d(depth*4, depth*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth*2)

        self.conv4 = nn.ConvTranspose2d(depth*2, depth, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(depth)
        
        self.conv5 = nn.ConvTranspose2d(depth, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.tanh(self.conv5(x))
        return x
    

class Discriminator(nn.Module):
    def __init__(self, depth=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, depth, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(depth)

        self.conv2 = nn.Conv2d(depth, depth*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth*2)

        self.conv3 = nn.Conv2d(depth*2, depth*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth*4)

        self.conv4 = nn.Conv2d(depth*4, depth*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(depth*8)
        
        self.conv5 = nn.Conv2d(depth*8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.bn2(x)
        x = self.lrelu(self.conv3(x))
        x = self.bn3(x)
        x = self.lrelu(self.conv4(x))
        x = self.bn4(x)
        x = self.conv5(x)
        return x

# DCGAN paper initializes using zero-centered Gaussian distribution, with standard deviation of 0.2
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)