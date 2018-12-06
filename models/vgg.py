'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, args=None, vgg_name="VGG19", freeze = False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.targets =  self._layer_detact(self.features, nn.MaxPool2d)
        self.args = args
        self.fc_out = True if args is None else self.args.fc_out
        if freeze:
            for p in self.parameters():
                    p.requires_grad=False
        self.out_dims = eval(self.args.out_dims)[-5:]
        self.fc_layers = self._make_fc()

    def _layer_detact(self, layers, target_layer):
        length = len(layers)
        return [i for i in range(length) if isinstance(layers[i], target_layer)]

    def _make_fc(self):
        if self.args.pool_out == "avg":
            layers = [
                nn.AdaptiveAvgPool1d(output) for output in self.out_dims
            ]
        elif self.args.pool_out == "max":
            layers = [
                nn.AdaptiveMaxPool1d(output) for output in self.out_dims
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_maps = []
        fc_layer = 0
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in self.targets:
                if self.fc_out:
                    out = self.fc_layers[fc_layer](x.view(x.size(0), x.size(1), -1))
                    if self.args.pool_out == "max":
                        out, _ = out.max(dim=1)
                    else:
                        out = out.mean(dim=1)
                    fc_layer += 1
                    feature_maps.append(torch.squeeze(out))
                else:
                    feature_maps.append(x.view(x.size(0), -1))
        out = x.view(x.size(0), -1)
        out = self.classifier(out)

        # out = F.softmax(out, dim=1)
        feature_maps[-1] = out
        return feature_maps

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11', freeze=True)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y)


# from loss import CrossEntropy
# test()
