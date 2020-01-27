import torch
import torch.nn
from training_utils import get_norm, get_padding, get_activ
from basic_modules import *


# outputs a number of arrays with discriminator scores is outputted
# arrays are of size 1 x S x S
# for each S in [dsc_scales] an array of size 1 x S x S is outputted

class BasePatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(BasePatchDiscriminator, self).__init__()
        self.dsc_channels = opt.dsc_channels
        self.dsc_layers = opt.dsc_layers
        self.dsc_scales = opt.dsc_scales
        self.norm_layer = get_norm(opt.dsc_norm)
        self.activ_layer = get_activ(opt.dsc_activ)
        self.padding_layer = get_padding(opt.dsc_padding)
        self.p_dropout = opt.dsc_dropout
        self.num_scales = len(self.dsc_scales)
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._dsc_layers = []
        self._patch_heads = []
        self._init_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=3,
                out_channels=self.dsc_channels,
                kernel_size=7,
                stride=1)
        ]

        self._dsc_layers = self._dsc_layers + self._init_layer

        factor = 1
        prev_factor = 1
        for i in range(self.dsc_layers):
            factor = factor * 2
            self._dsc_layers = self._dsc_layers + [DownSampleConv(in_channels=self.dsc_channels * prev_factor,
                                                                  out_channels=self.dsc_channels * factor,
                                                                  dropout=self.p_dropout,
                                                                  activ=opt.dsc_activ,
                                                                  norm=opt.dsc_norm,
                                                                  padding=opt.dsc_padding
                                                                  )]
            prev_factor = factor

        size = int(opt.load_size // factor) + 1

        for scale in self.dsc_scales:
            patch_head = [
                nn.Conv2d(in_channels=self.dsc_channels * factor,
                          out_channels=1,
                          kernel_size=size - scale,
                          padding=0,
                          stride=1),
                self.norm_layer(1),
                self.activ_layer()

            ]

            # wrapping because the scale head architecture is not fixed
            self._patch_heads = self._patch_heads + [patch_head]

        self._layers = nn.ModuleList(self._dsc_layers)

        for patch_head in self._patch_heads:
            for layer in patch_head:
                self._layers.append(layer)

    def forward(self, input):
        x = input
        for layer in self._dsc_layers:
            x = layer(x)
        y = []
        for patch_head in self._patch_heads:
            tmp_x = x
            for layer in patch_head:
                tmp_x = layer(tmp_x)
            y = y + [tmp_x]
        return y
