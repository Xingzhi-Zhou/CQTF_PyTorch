import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import copy
import sys
import os
import time

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../utils/')

from models.models_collection import *
from utils.tools import *
from config import *

# ############################################################
def get_current_file_name():
    try:
        # .py file name
        current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    except:
        try:
            # .ipynb in vscode 
            from IPython import get_ipython
            ip = get_ipython()
            if '__vsc_ipynb_file__' in ip.user_ns:
                file_name = ip.user_ns['__vsc_ipynb_file__']
                current_file_name = os.path.splitext(os.path.basename(file_name))[0]
        except:
            import ipynbname
            current_file_name = ipynbname.name()

    return current_file_name

# log and pth
file_name = get_current_file_name()
file_name = f'{file_name}_{TB}_{FB}'

log_dir = './log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
pth_dir = './pth/'
if not os.path.exists(pth_dir):
    os.makedirs(pth_dir)
log_file = os.path.join(log_dir, f'{file_name}')
pth_file = os.path.join(pth_dir, f'{file_name}.pth')

sys.stdout = Logger(log_file)

# ############################################################

# dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = datasets.CIFAR10(
    root='~/dataset/', 
    train=True, 
    download=True, 
    transform=transform_train
)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

testset = datasets.CIFAR10(root='~/dataset', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# model
Net = get_model(model_name)
model = Net()

if save_model:
    best_acc = 0.

if load_model:
    model.load_state_dict(torch.load(pth_file), strict=False)

model = model.to(device)

# train and test
criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, 
                            momentum = momentum,
                            weight_decay = weight_decay)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)

print('Model: {}， Total bits {}, Fractional bits {}'.format(model_name, TB, FB))

for epoch in range(epochs):

    # train

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0:
            # print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
            # pred = output.argmax(dim=1, keepdim=True)
            # correct = pred.eq(target.view_as(pred)).sum().item()
            # print("Accuracy:", correct / batch_size)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    
    lr_scheduler.step()

    # test

    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

    print(' * Loss {loss.avg:.4f}\t'
        'Prec@1 {top1.avg:.4f}'.format(
        loss=losses, top1=top1))
    
    if save_model:
        val_acc = top1.avg
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

if save_model:
    torch.save(best_model_wts, pth_file)
    print(f'Accuracy: {best_acc}, Model weights saved to {pth_file}')

