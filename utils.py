'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

from models import *

import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        if isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_model(args, config, device):
    teachers = []

    model_map = {"vgg19": VGG, "vgg19_BN": VGG, "resnet18": ResNet18, 'preactresnet18': PreActResNet18,
                 "googlenet": GoogLeNet, "densenet121": DenseNet121,"densenet_cifar": densenet_cifar,
                 "resnext": ResNeXt29_2x64d, "mobilenet": MobileNet, "dpn92": DPN92}

    # Add teachers models into teacher model list
    for t in args.teachers:
        if t in model_map:
            net = model_map[t](args)
            net.__name__ = t
            teachers.append(net)

    assert len(teachers) > 0, "teachers must be in %s" % " ".join(model_map.keys)

    # Initialize student model

    assert args.student in model_map, "students must be in %s" % " ".join(model_map.keys)
    student = model_map[args.student](args)

    # Model setup

    if device == "cuda":
        cudnn.benchmark = True

    for i, teacher in enumerate(teachers):
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device)
        if device == "cuda":
            teachers[i] = torch.nn.DataParallel(teacher)
            teachers[i].__name__ = teacher.__name__

    # Load parameters in teacher models
    for teacher in teachers:
        if teacher.__name__ != "shake_shake":
            checkpoint = torch.load('./checkpoint/%s/ckpt.t7' % teacher.__name__)
            model_dict = teacher.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            teacher.load_state_dict(model_dict)
            print("teacher %s acc: ", (teacher.__name__, checkpoint['acc']))

    student = student.to(device)
    if device == "cuda":
        out_dims = student.out_dims
        student = torch.nn.DataParallel(student)
        student.out_dims = out_dims

    if args.teacher_eval:
        for teacher in teachers:
            teacher.eval()

    return teachers, student
