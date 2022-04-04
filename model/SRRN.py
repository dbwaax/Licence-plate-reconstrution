import torch 
import torch.nn as nn
from  model.ResBlock import ResBlock
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),    
            nn.ReLU()        
        )
        self.in_channels = 64
        self.resnet = nn.Sequential(
            self.make_layer(ResBlock,64,2,stride=1),
            self.make_layer(ResBlock,128,2,stride=2),
            self.make_layer(ResBlock,256,2,stride=2),
            self.make_layer(ResBlock,512,2,stride=2),
        )

        self.Deconv_block = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,3,kernel_size=(4,2),stride=1),
            nn.Tanh(),
        )

        self._init_weights(self.pre)
        self._init_weights(self.resnet)
        self._init_weights(self.Deconv_block)
    def make_layer(self,block,channels,num_blocks,stride):
        strides = [stride] + [1] *(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,channels,stride))
            self.in_channels = channels
        return nn.Sequential(*layers)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.LeakyReLU,nn.ConvTranspose2d,nn.Tanh,nn.ReLU)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 1.0)
    def forward(self,x):
        out = self.pre(x)
        out = self.resnet(out)
        out = self.Deconv_block(out)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2),          
        )

        self.end = nn.Sequential(
            nn.Conv2d(256,1,(1,5),stride=1),
            nn.Sigmoid()
        )
        self._init_weights(self.pre)
        self._init_weights(self.middle)
        self._init_weights(self.end)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.LeakyReLU,nn.BatchNorm2d,nn.AvgPool2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 1.0)
    def forward(self, x):                             #64x128x3
        x = self.pre(x)
        x = self.middle(x)
        x = self.end(x)
#         x = x.view()
        return x      


if __name__ == "__main__":
    t = torch.randn(2,3,24,94)
    # print(t)
    d1 = Generator()
    d2 = Discriminator()
    r1 = d1(t)
    r2 = d2(t)
    print(r1.squeeze().size())

