
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        return (-grad_output)

def grad_reverse(x):
    return GradReverse()(x)

class Discriminator(nn.Module):
    def __init__(self, outputs_size, K = 2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv2 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv3 = nn.Conv2d(in_channels=outputs_size, out_channels=2, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = x[:,:,None,None]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        return out

class Discriminators(nn.Module):
    def __init__(self, output_dims, grl):
        super(Discriminators, self).__init__()
        self.discriminators = [Discriminator(i) for i in output_dims]
        self.grl = grl
    
    def forward(self, x):
        if self.grl == True:
            out = [self.discriminators[i](grad_reverse(x[i])) for i in range(len(self.discriminators))]
        else:
            out = [self.discriminators[i](x[i]) for i in range(len(self.discriminators))]

        return out
