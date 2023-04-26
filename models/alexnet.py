import torch
import torch.nn as nn
import torch.nn.init as init
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from qconv import QConv2d
from qlinear import QLinear
from utils import View

__all__ = ["alexnet", "alexnet_cifar"]


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(96, 256, 5, 1, 2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(256, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(384, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            QConv2d(384, 256, 3, 1, 1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            View(256 * 6 * 6),
            QLinear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            # nn.ReLU(inplace=True),
            nn.Identity(),
            # nn.Dropout() if dropout else None,
            QLinear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout() if dropout else None,
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)

class AlexNet_Cifar(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_Cifar, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(3, 96, 5, 1, 2, bias=False), # 32 => 32 
                nn.MaxPool2d(kernel_size=2, stride=2), # 32 => 16
                nn.BatchNorm2d(96),
                # nn.ReLU(inplace=True),
                nn.Identity(),
                QConv2d(96, 256, 5, 1, 2, bias=False), # 16 => 16
                nn.MaxPool2d(kernel_size=2, stride=2), # 16 => 8
                nn.BatchNorm2d(256),
                # nn.ReLU(inplace=True),
                nn.Identity(),
                QConv2d(256, 384, 3, 1, 1, bias=False), # 8 => 8
                nn.BatchNorm2d(384),
                # nn.ReLU(inplace=True),
                nn.Identity(),
                QConv2d(384, 384, 3, 1, 1, bias=False), # 8 => 8
                nn.BatchNorm2d(384),
                # nn.ReLU(inplace=True),
                nn.Identity(),
                QConv2d(384, 256, 3, 1, 1, bias=False), # 8 => 8
                nn.MaxPool2d(kernel_size=2, stride=2), # 4 => 4
                nn.BatchNorm2d(256),
                # nn.ReLU(inplace=True),
                nn.Identity(),
                View(256 * 4 * 4), # 256 * 4 * 4
                QLinear(256 * 4 * 4, 1024, bias=False), # 256 * 4 * 4 => 1024
                nn.BatchNorm1d(1024),
                # nn.ReLU(inplace=True),
                nn.Identity(),
                # nn.Dropout() if dropout else None,
                QLinear(1024, 1024, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                # nn.Dropout() if dropout else None,
                nn.Linear(1024, num_classes), # 1024 => num_classes
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)

def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model

def alexnet_cifar(**kwargs):
    model = AlexNet_Cifar(**kwargs)
    return model


if __name__ == '__main__':
    m = alexnet()
    m_cifar = alexnet_cifar(num_classes=10)
    print(m)
    print("imagenet shape inference (2, 3, 224, 224)")
    x = torch.randn(2, 3, 224, 224)
    print(m(x).shape)
    print("\n\n")
    print(m_cifar)
    print("CIFAR10 shape inference (2, 3, 32, 32)")
    x = torch.randn(2, 3, 32, 32)
    print(m_cifar(x).shape)
    
