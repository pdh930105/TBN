import argparse
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

from qconv import QConv2d, TBNConv2d
from qlinear import QLinear, TBNLinear
from train import validate, AverageMeter

from utils import hardware_evaluation

def main():
    parser = argparse.ArgumentParser(description='NeuroSim inference')
    parser.add_argument('--checkpoint', required=True, help='path to checkpoint')
    parser.add_argument('--neurosim', action='store_true', help='use NeuroSim inference log')

    args = parser.parse_args()
    neurosim = args.neurosim

    checkpoint = torch.load(args.checkpoint)
    args = checkpoint['args']
    args.neurosim = neurosim
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    TBNConv2d.qa_config = TBNLinear.qa_config = {
        'delta': args.ternary_delta,
        'order': args.ternary_order,
        'momentum': args.ternary_momentum,
        'scale': bool(args.ternary_no_scale)
    }
    print('==> Ternary Config:', TBNConv2d.qa_config)

    # Data loading code
    # Normalize, but do not standardize, the data
    normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # create dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        val_dataset = datasets.CIFAR10(args.data_path, train=False, download=True, transform=test_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    elif args.dataset == 'cifar100':
        args.num_classes = 100
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = datasets.CIFAR100(args.data_path, train=False, download=True, transform=test_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    else:
        valdir = os.path.join(args.data_path, 'val')
        args.num_classes = 1000

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # create model
    print(f"load model : {args.arch}")
    model = models.__dict__[args.arch](num_classes=args.num_classes, args=args)
    model.load_state_dict(checkpoint['state_dict'])

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    top1, top5, loss = validate(val_loader, model, criterion, args)
    print(f"top1 : {top1}, top5 : {top5}, loss : {loss}")


    if args.neurosim:
        print("tried neurosim inference result")
        print(f" SubArray : {args.subArray} \n ADCprecision : {args.ADCprecision} \n ADC mode : {args.adc_mode} \n Cell Precision : {args.cellBit}")
        model = model.eval()
        for i, (data, target) in enumerate(val_loader):
            if i == 0:
                hook_handle_list = hardware_evaluation(model, args.wl_weight, args.wl_activate, args.arch, None)
                with torch.no_grad():
                    data= data.cuda(args.gpu)
                    output = model(data)
            

if __name__ == '__main__':
    main()



