'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
import random

import os
import argparse
import numpy as np

from models import *
from models import discriminator
from utils import progress_bar, get_model
from loss import *

# ================= Arugments ================ #

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--d_lr', default=0.1, type=float, help='discriminator learning rate')
parser.add_argument('--teachers', default='[\'shufflenetg2\']', type=str, help='teacher networks type')
parser.add_argument('--student', default='shufflenetg2', type=str, help='student network type')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu_id', default='3', type=str, help='gpu id')
parser.add_argument('--gamma', default='[1,1,1,1,1]', type=str, help='')
parser.add_argument('--eta', default='[1,1,1,1,1]', type=str, help='')
parser.add_argument('--fc_out', default=1, type=int, help='if immediate output from fc-layer')
parser.add_argument('--loss', default="ce", type=str, help='loss selection')
parser.add_argument('--adv', default=1, type=int, help='add discriminator or not')
parser.add_argument('--name', default=None, type=str, help='the name of this experiment')
parser.add_argument('--pool_out', default="max", type=str, help='the type of pooling layer of output')
parser.add_argument('--out_layer', default="[-1]", type=str, help='the type of pooling layer of output')
parser.add_argument('--out_dims', default="[5000,1000,500,200,10]", type=str, help='the dims of output pooling layers')
parser.add_argument('--teacher_eval', default=0, type=int, help='use teacher.eval() or not')

# model config
parser.add_argument('--depth', type=int, default=26)
parser.add_argument('--base_channels', type=int, default=96)
parser.add_argument('--grl', type=bool, default=False, help="gradient reverse layer")

# run config
parser.add_argument('--outdir', type=str, default="results")
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--num_workers', type=int, default=7)

# optim config
parser.add_argument('--epochs', type=int, default=1800)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--lr_min', type=float, default=0)


args = parser.parse_args()

# ================= Config Collection ================ #

model_config = OrderedDict([
    ('depth', args.depth),
    ('base_channels', args.base_channels),
    ('input_shape', (1, 3, 32, 32)),
    ('n_classes', 10),
    ('out_dims', args.out_dims),
    ('fc_out', args.fc_out),
    ('pool_out', args.pool_out)
])

optim_config = OrderedDict([
    ('epochs', args.epochs),
    ('batch_size', args.batch_size),
    ('base_lr', args.base_lr),
    ('weight_decay', args.weight_decay),
    ('momentum', args.momentum),
    ('nesterov', args.nesterov),
    ('lr_min', args.lr_min),
])

data_config = OrderedDict([
    ('dataset', 'CIFAR10'),
])

run_config = OrderedDict([
    ('seed', args.seed),
    ('outdir', args.outdir),
    ('num_workers', args.num_workers),
])

config = OrderedDict([
    ('model_config', model_config),
    ('optim_config', optim_config),
    ('data_config', data_config),
    ('run_config', run_config),
])

print(args)

# ================= Initialization ================ #

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
device = 'cuda'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# ================= Data Loader ================ #

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

if args.student == "densenet_cifar":
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    print("batch_size =", 64)

else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

optim_config['steps_per_epoch'] = len(trainloader)

# ================= Model Setup ================ #

args.teachers = eval(args.teachers)

print('==> Training', args.student if args.name is None else args.name)
print('==> Building model..')

# get models as teachers and students
teachers, student = get_model(args, config, device="cuda")

print("==> Teacher(s): ", " ".join([teacher.__name__ for teacher in teachers]))
print("==> Student: ", args.student)

dims = [student.out_dims[i] for i in eval(args.out_layer)]
print("dims:", dims)

update_parameters = [{'params': student.parameters()}]

if args.adv:
    discriminators = discriminator.Discriminators(dims, grl=args.grl)
    for d in discriminators.discriminators:
        d = d.to(device)
        if device == "cuda":
            d = torch.nn.DataParallel(d)
        update_parameters.append({'params': d.parameters(), "lr": args.d_lr})

print(args)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s-generator/ckpt.t7' % "_".join(args.teachers))
    student.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

# ================= Loss Function for Generator ================ #

if args.loss == "l1":
    loss = F.l1_loss
elif args.loss == "l2":
    loss = F.mse_loss
elif args.loss == "l1_soft":
    loss = L1_soft
elif args.loss == "l2_soft":
    loss = L2_soft
elif args.loss == "ce":
    loss = CrossEntropy      # CrossEntropy for multiple classification

criterion = betweenLoss(eval(args.gamma), loss=loss)

# ================= Loss Function for Discriminator ================ #

if args.adv:
    discriminators_criterion = discriminatorLoss(discriminators, eval(args.eta))
else:
    discriminators_criterion = discriminatorFakeLoss()

# ================= Optimizer Setup ================ #

if args.student == "densenet_cifar":
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(2, len(teachers)), 250 * min(2, (len(teachers)))],gamma=0.1)
    print("nesterov = True")
elif args.student == "mobilenet":
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)
else:
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)

# ================= Training and Testing ================ #

def teacher_selector(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx]

def output_selector(outputs, answers, idx):
    return [outputs[i] for i in idx], [answers[i] for i in idx]

def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    discriminator_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        total += targets.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Get output from student model
        outputs = student(inputs)
        # Get teacher model
        teacher = teacher_selector(teachers)
        # Get output from teacher model
        answers = teacher(inputs)
        # Select output from student and teacher
        outputs, answers = output_selector(outputs, answers, eval(args.out_layer))
        # Calculate loss between student and teacher
        loss = criterion(outputs, answers)
        # Calculate loss for discriminators
        d_loss = discriminators_criterion(outputs, answers)
        # Get total loss
        total_loss = loss + d_loss

        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        discriminator_loss += d_loss.item()
        _, predicted = outputs[-1].max(1)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Teacher: %s | Lr: %.4e | G_Loss: %.3f | D_Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (teacher.__name__, scheduler.get_lr()[0], train_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    student.eval()
    test_loss = 0
    correct = 0
    total = 0
    discriminator_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            total += targets.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            # Get output from student model
            outputs = student(inputs)
            # Get teacher model
            teacher = teacher_selector(teachers)
            # Get output from teacher model
            answers = teacher(inputs)
            # Select output from student and teacher
            outputs, answers = output_selector(outputs, answers, eval(args.out_layer))
            # Calculate loss between student and teacher
            loss = criterion(outputs, answers)
            # Calculate loss for discriminators
            d_loss = discriminators_criterion(outputs, answers)

            test_loss += loss.item()
            discriminator_loss += d_loss.item()
            _, predicted = outputs[-1].max(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Lr: %.4e | G_Loss: %.3f | D_Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (scheduler.get_lr()[0], test_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        best_acc = max(100. * correct / total, best_acc)

    # Save checkpoint (the best accuracy).
    if epoch % 10 == 0 and best_acc == (100. * correct / total):
        print('Saving..')
        state = {
            'net': student.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        FILE_PATH = './checkpoint' + '/' + "_".join(args.teachers) + '-generator'
        if os.path.isdir(FILE_PATH):
            # print 'dir exists'generator
            pass
        else:
            # print 'dir not exists'
            os.mkdir(FILE_PATH)
        save_name = './checkpoint' + '/' + "_".join(args.teachers) + '-generator/ckpt.t7'
        torch.save(state, save_name)

for epoch in range(start_epoch, start_epoch+args.epochs*(len(teachers))):
    train(epoch)
    test(epoch)
