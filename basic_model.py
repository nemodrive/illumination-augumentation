import os
from collections import OrderedDict
from training_utils import *
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.trainable = opt.trainable
        if self.gpu_ids:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name + \
                                     '_' + opt.dsc_model + \
                                     '_' + opt.enc_model + \
                                     '_' + opt.dec_model)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.gradient_norm_names = []
        self.metric = 0

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    def setup(self, opt):
        if self.trainable:
            self.schedule = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.trainable or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedule:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def get_current_gradient_norms(self):
        gnorms_ret = OrderedDict()
        for name in self.gradient_norm_names:
            if isinstance(name, str):
                gnorms_ret[name] = float(getattr(self, 'gnorm_' + name))
        return gnorms_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def print_networks(self, verbose):
        print('\t\t\t --- Networks intilizaed --- \t\t\t')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params = num_params + param.numel()
                if verbose:
                    print(net)
                print('%s has %.3f M trainable parameters' % (name, num_params / 1e6))
        print('\t\t\t --------------------------- \t\t\t')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
