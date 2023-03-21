import math
import torch.nn.functional as F


def lcm(*list):
    minimum = 1
    for i in list:
        minimum = int(i) * int(minimum) / math.gcd(int(i), int(minimum))
    return int(minimum)