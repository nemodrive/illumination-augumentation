import torch
import torch.
import torch.nn as nn
import torch.nn.functional as F


USE_PYTORCH_IN = True

class IdentityNorm(nn.Module):
    def __init__(self):
        super(IdentityNorm, self).__init__()

    def forward(self, input):
        return input

class AdaINNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaINNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var  = self.running_var.repeat(c)

        x_reshaped = x.contiguos().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True,
                           self.momentum, self.eps)

        return out.view(b, c *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'



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