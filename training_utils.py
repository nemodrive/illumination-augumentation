import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
import torch.optim as optim
from basic_layers import *


def get_padding(type='zeros'):
    if type == 'zeros':
        pad_layer = partial(nn.ZeroPad2d)
    elif type == 'reflection':
        pad_layer = partial(nn.ReflectionPad2d)
    elif type == 'replicate':
        pad_layer = partial(nn.ReplicationPad2d)
    elif type == 'none':
        pad_layer = None
    return pad_layer


def get_norm(type='instance'):
    if type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif type == 'instance':
        norm_layer = partial(InstanceNorm2d, affine=False)
    else:
        norm_layer = partial(IdentityNorm)
    return norm_layer


def get_activ(type='relu'):
    if type == 'relu':
        activ_layer = partial(nn.ReLU, inplace=True)
    elif type == 'leaky_relu':
        activ_layer = partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
    elif type == 'elu':
        activ_layer = partial(nn.ELU, alpha=0.2, inplace=True)
    elif type == 'gelu':
        activ_layer = partial(nn.GELU)
    return activ_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            factor = 1.0 - epoch / 1000.
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=opt.decay_steps, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, threshold=0.01,
                                                         patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.decay_steps, eta_min=0)
    return scheduler


def init_weights(network, type='normal', gain=0.02, bias=0.):
    def init_function(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if type == 'normal':
                init.normal_(model.weight.data, mean=0., std=gain)
            elif type == 'xavier':
                init.xavier_normal_(model.weight.data, gain=gain)
            elif type == 'kaiming':
                init.kaiming_normal_(model.weight.data, a=0, mode='fai_in', nonlinearity='leaky_relu')
            elif type == 'orthogonal':
                init.orthogonal_(model.weight.data, gain=gain)
            if hasattr(model, 'bias') and model.bias is not None:
                init.constant_(model.bias.data, bias)
            elif classname.find('Batchnorm2d') != -1:
                init.normal_(model.weight.data, mean=1., std=gain)
                init.constant_(model.bias.data, gain)

    print('initialized network with %s' % type)
    network.apply(init_function)


def init_model(model, type='normal', gain=0.02, bias=0., gpu_ids=[]):
    if len(gpu_ids):
        assert (torch.cuda.is_available())
        model.to(gpu_ids[0])
        # model = torch.nn.DataParallel(model, gpu_ids)
    init_weights(model, type, gain, bias)
    return model
