from basicsr.utils.registry import ARCH_REGISTRY

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

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