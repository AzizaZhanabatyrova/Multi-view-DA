import torch.nn as nn
import torch.nn.functional as F
        
        
class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64, extra_layers = 0):
        super(FCDiscriminator, self).__init__()
        
        self.extra_layers = extra_layers

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)

        last_multiplier = 8

        if extra_layers > 0:
            self.conv = []
            last_multiplier = 2**(extra_layers+3)
            for i in range(0, extra_layers):
                self.conv.append(nn.Conv2d(ndf*(2**(i+3)), ndf*(2**(i+4)), kernel_size=4, stride=2, padding=1).cuda())

        self.classifier = nn.Conv2d(ndf*last_multiplier, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)

        if self.extra_layers > 0:
            for i in range(0, self.extra_layers):
                x = self.conv[i](x)
                x = self.leaky_relu(x)

        x = self.classifier(x)

        return x
