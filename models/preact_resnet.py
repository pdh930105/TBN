import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
name = 0

from qconv import TBNConv2d, QConv2d
from qlinear import TBNLinear, QLinear

__all__ = ["PreActResNet", "preact_resnet_18", "preact_resnet_34", "preact_resnet_50", "preact_resnet_18_cifar", "preact_resnet_50_cifar", "preact_resnet_18_cifar_q", "preact_resnet_50_cifar_q"]


def tbnconv3x3(in_planes, out_planes, stride=1, padding=1, bias=False, args=None):
    """3x3 convolution with padding"""
    global name
    conv = TBNConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias, inference=args.inference,subArray= args.subArray, adc_mode=args.adc_mode, ADCprecision=args.ADCprecision,
                     name = 'TBNConv3x3'+'_'+str(name)+'_')

    name +=1
    return conv

def tbnconv1x1(in_planes, out_planes, stride=1, bias=False, args=None):
    """1x1 convolution"""
    global name
    conv = TBNConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias, inference=args.inference, subArray = args.subArray, adc_mode=args.adc_mode, ADCprecision=args.ADCprecision,
                     name = 'TBNConv3x3'+'_'+str(name)+'_')
    name +=1
    return conv

def qconv(in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, args=None):
    """qconv"""
    global name
    # future change TBNConv2d to QConv2d
    conv = QConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, logger=args.logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,
                         wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision, adc_mode=args.adc_mode, vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                         name = 'QConv'+'_'+str(name)+'_', model = args.model)
    name +=1
    return conv

def qlinear(in_planes, out_planes, args):
    global name
    linear = QLinear(in_planes, out_planes, 
                        logger=args.logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision, adc_mode=args.adc_mode, vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, 
                        name='FC'+'_'+str(name)+'_', model = args.model)
    return linear


class ShortCutA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = F.adaptive_avg_pool2d(x, y.shape[2:])
        x = self.relu(self.bn(x))
        if x.size(1) == y.size(1):
            return x + y
        elif x.size(1) > y.size(1):
            x[:, :y.size(1), ...] += y
            return x
        else:
            y[:, :x.size(1), :, :] += x
            return y


class ShortCutB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.conv(self.relu(self.bn(x))) + y


class ShortCutC(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = F.adaptive_avg_pool2d(x, y.shape[2:])
        return self.conv(self.relu(self.bn(x))) + y
    
class ShortCutQ(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, args=None):
        super().__init__()
        self.stride= stride
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)
        self.conv = qconv(in_planes, out_planes, 1, stride=stride, padding=0, bias=False, args=args)

    def forward(self, x, y):
        return self.conv(self.relu(self.bn(x))) + y


def ShortCut(x: torch.Tensor, y: torch.Tensor):
    return x + y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut, args=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.conv1 = tbnconv3x3(in_planes, planes, stride, args=args)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.conv2 = tbnconv3x3(planes, planes, 1, args=args)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = self.downsample(identity, x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut, args=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.conv1 = tbnconv1x1(in_planes, planes, 1, args=args)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.conv2 = tbnconv3x3(planes, planes, stride=stride, args=args)

        self.bn3 = nn.BatchNorm2d(planes)
        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.Identity()
        self.conv3 = tbnconv1x1(planes, planes * 4, 1, args=args)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = self.conv3(self.relu3(self.bn3(x)))
        x = self.downsample(identity, x)
        return x


class BasicBlockR(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut, args=None):
        super(BasicBlockR, self).__init__()
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = tbnconv3x3(in_planes, planes, stride, args=args)

        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = tbnconv3x3(planes, planes, args=args)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.bn1(self.relu1(x)))
        x = self.conv2(self.bn2(self.relu2(x)))
        x = self.downsample(identity, x)
        return x


class BottleneckR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, args=None):
        super(BottleneckR, self).__init__()
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = tbnconv1x1(in_planes, planes, args=args)

        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = tbnconv3x3(planes, planes, stride=stride, args=args)

        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.Identity()
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.conv3 = tbnconv1x1(planes, planes * 4,args=args)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(self.bn1(self.relu1(x)))
        x = self.conv2(self.bn2(self.relu2(x)))
        x = self.conv3(self.bn3(self.relu3(x)))
        x = self.downsample(identity, x)
        return x


class BasicBlockBi(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=ShortCut):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.Identity()
        self.conv1 = TBNConv2d(in_planes, planes, 3, stride, 1)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.Identity()
        self.conv2 = TBNConv2d(planes, planes, 3, 1, 1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.downsample(x, self.conv1(self.relu1(self.bn1(x))))
        x = x + self.conv2(self.relu2(self.bn2(x)))
        return x


class PreActResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, shortcut=ShortCutC, small_stem=False, args=None):
        self.in_planes = 64
        super(PreActResNet, self).__init__()
        if small_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.Identity(),
                TBNConv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.Identity(),
                TBNConv2d(32, self.in_planes, 3, 1, 1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, shortcut, 64, layers[0], args=args)
        self.layer2 = self._make_layer(block, shortcut, 128, layers[1], args=args,stride=2)
        self.layer3 = self._make_layer(block, shortcut, 256, layers[2], args=args,stride=2)
        self.layer4 = self._make_layer(block, shortcut, 512, layers[3], args=args,stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, shortcut, planes, blocks, args, stride=1):
        downsample = ShortCut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = shortcut(self.in_planes, planes * block.expansion, stride)
        layers = [block(self.in_planes, planes, stride, downsample, args=args)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

class PreActResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, shortcut=ShortCutC, small_stem=False,  args=None):
        self.in_planes = 64
        super(PreActResNet_CIFAR, self).__init__()
        num_classes = args.num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # 7 -> 3 for CIFAR
        self.layer1 = self._make_layer(block, shortcut, 64, layers[0], args=args)
        self.layer2 = self._make_layer(block, shortcut, 128, layers[1], stride=2, args=args)
        self.layer3 = self._make_layer(block, shortcut, 256, layers[2], stride=2, args=args)
        self.layer4 = self._make_layer(block, shortcut, 512, layers[3], stride=2, args=args)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, shortcut, planes, blocks, stride=1, args=None):
        downsample = ShortCut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = shortcut(self.in_planes, planes * block.expansion, stride)
        layers = [block(self.in_planes, planes, stride, downsample, args=args)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


class PreActResNet_CIFAR_Q(nn.Module):
    def __init__(self, block, layers, num_classes=10, shortcut=ShortCutQ, small_stem=False,  args=None):
        self.in_planes = 64
        super(PreActResNet_CIFAR_Q, self).__init__()
        num_classes = args.num_classes
        self.conv1 = qconv(3, 64, kernel_size=3, stride=1, padding=1, args=args) # 7 -> 3 for CIFAR
        self.layer1 = self._make_layer(block, shortcut, 64, layers[0], args=args)
        self.layer2 = self._make_layer(block, shortcut, 128, layers[1], stride=2, args=args)
        self.layer3 = self._make_layer(block, shortcut, 256, layers[2], stride=2, args=args)
        self.layer4 = self._make_layer(block, shortcut, 512, layers[3], stride=2, args=args)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qlinear(512 * block.expansion, num_classes, args=args)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, shortcut, planes, blocks, stride=1, args=None):
        downsample = ShortCut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = shortcut(self.in_planes, planes * block.expansion, stride, args=args)
        layers = [block(self.in_planes, planes, stride, downsample, args=args)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x



model_list = {
    "18": (BasicBlock, [2, 2, 2, 2]),
    "34": (BasicBlock, [3, 4, 6, 3]),
    "50": (Bottleneck, [3, 4, 6, 3]),
    "101": (Bottleneck, [3, 4, 23, 3]),
    "152": (Bottleneck, [3, 8, 36, 3]),
    "18r": (BasicBlockR, [2, 2, 2, 2]),
    "34r": (BasicBlockR, [3, 4, 6, 3]),
    "50r": (BottleneckR, [3, 4, 6, 3]),
    "101r": (BottleneckR, [3, 4, 23, 3]),
    "152r": (BottleneckR, [3, 8, 36, 3]),
    '18bi': (BasicBlockBi, [2, 2, 2, 2]),
    "34bi": (BasicBlockBi, [3, 4, 6, 3]),
}

shortcut_list = {
    'A': ShortCutA,
    'B': ShortCutB,
    'C': ShortCutC,
    'Q': ShortCutQ,
}


def preact_resnet(depth="18", shortcut='A', args='None',**kwargs):
    depth = str(depth)
    assert depth in model_list.keys(), "Only support depth={" + ",".join(map(str, model_list.keys())) + "}"
    assert shortcut in shortcut_list.keys(), "Only support shortcut={" + ",".join(shortcut_list.keys()) + "}"
    return PreActResNet(*model_list[depth], shortcut=shortcut_list[shortcut], args=args,**kwargs)


def preact_resnet_18(args=None, **kwargs):
    return PreActResNet(*model_list['18'], shortcut=shortcut_list['C'], args=args, **kwargs)


def preact_resnet_34(args=None, **kwargs):
    return PreActResNet(*model_list['34'], shortcut=shortcut_list['C'], args=args,**kwargs)


def preact_resnet_50(args=None, **kwargs):
    return PreActResNet(*model_list['50'], shortcut=shortcut_list['A'], args=args,**kwargs)

def preact_resnet_18_cifar(args=None, **kwargs):
    return PreActResNet_CIFAR(*model_list['18'], shortcut=shortcut_list['C'], args=args,**kwargs)

def preact_resnet_18_cifar_q(args=None, **kwargs):
    return PreActResNet_CIFAR_Q(*model_list['18'], shortcut=shortcut_list['Q'], args=args,**kwargs)

def preact_resnet_50_cifar(args=None, **kwargs):
    return PreActResNet_CIFAR(*model_list['50'], shortcut=shortcut_list['A'], args=args, **kwargs)


def preact_resnet_50_cifar_q(args=None, **kwargs):
    return PreActResNet_CIFAR_Q(*model_list['50'], shortcut=shortcut_list['Q'], args=args, **kwargs)


if __name__ == '__main__':
    import torch
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='preact_resnet_18_cifar_q', type=str, help='model name')
    parser.add_argument('--wl_weight', default=8, type=int, help='weight bit width')
    parser.add_argument('--wl_activate', default=8, type=int, help='activation bit width')
    parser.add_argument('--wl_error', default=8, type=int, help='error bit width')
    parser.add_argument('--inference', default=0, help='run hardware inference simulate')
    parser.add_argument('--subArray', default=128, help='size of subArray (e.g. 128*128)')
    parser.add_argument('--ADCprecision', default=5, help='ADC precision (e.g. 5-bit)')
    parser.add_argument('--adc_mode', default='original', help='ADC mode (e.g. original, linear, none)')
    parser.add_argument('--cellBit', default=4, help='cell precision (e.g. 4-bit/cell)')
    parser.add_argument('--onoffratio', default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
    # if do not run the device retention / conductance variation effects, set vari=0, v=0
    parser.add_argument('--vari', default=0, help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
    parser.add_argument('--t', default=0, help='retention time')
    parser.add_argument('--v', default=0, help='drift coefficient')
    parser.add_argument('--detect', default=0, help='if 1, fixed-direction drift, if 0, random drift')
    parser.add_argument('--target', default=0, help='drift target for fixed-direction drift')

    args = parser.parse_args()
    args.logger = None
    args.num_classes = 10

    if args.model == 'preact_resnet_18_cifar_q':
        model = preact_resnet_18_cifar_q(args=args)
    elif args.model == 'preact_resnet_50_cifar_q':
        model = preact_resnet_50_cifar_q(args=args)

    dummy_input = torch.randn(1, 3, 32, 32)
    print(model(dummy_input))
