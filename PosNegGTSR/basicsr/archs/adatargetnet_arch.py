from basicsr.utils.registry import ARCH_REGISTRY

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

@ARCH_REGISTRY.register()
class RRDBAdaNet(nn.Module):
    def __init__(self, in_nc = 3, out_nc = 3, nf = 64, nb = 23, gc=32):
        super(RRDBAdaNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        # self.conv_zero = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.nnup2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, x_feat=None, out_feat=False):
        if x_feat is not None:
            fea = x_feat
        else:
            fea = self.conv_first(x)

        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        if out_feat:
            return fea

            # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        # _, _, H, W = fea.size()
        # fea = self.lrelu(self.upconv1(F.upsample_nearest(fea, [360,640])))
        # fea = self.lrelu(self.upconv2(F.upsample_nearest(fea, [720,1280])))
        fea = self.nnup2(fea)
        fea = self.upconv1(fea)
        fea = self.lrelu(fea)

        fea = self.nnup2(fea)
        fea = self.upconv2(fea)
        fea = self.lrelu(fea)

        # fea = self.lrelu(self.upconv1(F.upsample(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.upsample(fea, scale_factor=2, mode='nearest')))
        out_f = self.HRconv(fea)
        out = self.conv_last(self.lrelu(out_f))

        return out

@ARCH_REGISTRY.register()
class LocNet(torch.nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()

        ch = 9**2 *3 + 7**2 *3
        self.layer1 = nn.Linear(ch, ch*2)
        self.bn1 = nn.BatchNorm1d(ch*2)
        self.layer2 = nn.Linear(ch*2, ch*2)
        self.bn2 = nn.BatchNorm1d(ch*2)
        self.layer3 = nn.Linear(ch*2, ch)
        self.bn3 = nn.BatchNorm1d(ch)
        self.layer4 = nn.Linear(ch, 6)

        # Init weights
        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.lower().find('conv') != -1:
        #         # print(classname)
        #         nn.init.kaiming_normal(m.weight)
        #         nn.init.constant(m.bias, 0)
        #     elif classname.find('bn') != -1:
        #         m.weight.data.normal_(1.0, 0.02)
        #         m.bias.data.fill_(0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.layer4(x)
        return x