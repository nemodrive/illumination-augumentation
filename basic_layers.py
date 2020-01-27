import torch
import torch.nn as nn
import torch.nn.functional as F


USE_PYTORCH_IN = False

class IdentityNorm(nn.Module):
    def __init__(self):
        super(IdentityNorm, self).__init__()

    def forward(self, input):
        return input


class InstanceNorm(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        self.scale = torch.nn.Parameter(torch.Tensor(num_features))
        self.shift = torch.nn.Parameter(torch.Tensor(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.scale.data.normal_(mean=0., std=0.02)
            self.shift.data.zero_()

    def forward(self, input):
        size = input.size()
        x = input.view(size[0], size[1], size[2] * size[3])
        mean = x.mean(2, keepdim=True)
        x = x - mean
        std = torch.rsqrt((x ** 2).mean(2, keepdim=True) + self.eps)
        norm_features = (x * std).view(*size)

        if self.affine:
            output = norm_features * self.scale[:, None, None] + self.shift[:, None, None]
        else:
            output = norm_features

        return output

InstanceNorm2d = nn.InstanceNorm2d if USE_PYTORCH_IN else InstanceNorm