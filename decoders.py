import torch
import torch.nn
from training_utils import get_norm, get_padding, get_activ
from basic_modules import *


class BaseDecoder(nn.Module):
    def __init__(self, opt, out_channels, type='continous'):
        super(BaseDecoder, self).__init__()
        self.out_channels = out_channels
        self.enc_channels = opt.enc_channels
        self.dec_channels = opt.dec_channels
        self.num_upsamples = opt.num_downsamples
        # these are layer constructors, not implicit layers
        self.norm_layer = get_norm(opt.dec_norm)
        self.activ_layer = get_activ(opt.dec_activ)
        self.padding_layer = get_padding(opt.dec_padding)
        self.p_dropout = opt.dec_dropout
        if type == 'continous':
            self.activ_final = nn.Tanh()
        if type == 'probabilities':
            self.activ_final = nn.Sigmoid()
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []

        factor = 2 ** self.num_upsamples

        # going from encoding channel space to decoding channel space
        self._layers = self._layers + [UpSampleConv(in_channels=self.enc_channels * factor,
                                                    out_channels=self.dec_channels * factor,
                                                    dropout=self.p_dropout,
                                                    activ=opt.dec_activ,
                                                    norm=opt.dec_norm,
                                                    padding='zeros'
                                                    )]
        prev_factor = factor

        for i in range(self.num_upsamples - 1):
            factor = int(factor / 2)
            self._layers = self._layers + [UpSampleConv(in_channels=self.dec_channels * prev_factor,
                                                        out_channels=self.dec_channels * factor,
                                                        dropout=self.p_dropout,
                                                        activ=opt.dec_activ,
                                                        norm=opt.dec_norm,
                                                        padding='zeros'
                                                        )]
            prev_factor = factor

        self._final_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=self.dec_channels * factor,
                out_channels=self.out_channels,
                kernel_size=7,
                stride=1),
            self.activ_final
        ]

        self._layers = self._layers + self._final_layer

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class BaseDilationDecoder(nn.Module):
    def __init__(self, opt, out_channels, type='continous'):
        super(BaseDilationDecoder, self).__init__()
        self.out_channels = out_channels
        self.enc_channels = opt.enc_channels
        self.dec_channels = opt.dec_channels
        self.num_upsamples = opt.num_downsamples
        self.dilations = opt.dec_dilations
        # these are layer constructors, not implicit layers
        self.norm_layer = get_norm(opt.dec_norm)
        self.activ_layer = get_activ(opt.dec_activ)
        self.padding_layer = get_padding(opt.dec_padding)
        self.p_dropout = opt.dec_dropout
        if type == 'continous':
            self.activ_final = nn.Tanh()
        if type == 'probabilities':
            self.activ_final = nn.Sigmoid()
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []

        factor = 2 ** self.num_upsamples
        prev_factor = factor

        self._layers = self._layers + [MultiDilatedConv(in_channels=self.dec_channels * factor,
                                                        out_channels=self.dec_channels * factor,
                                                        dil_channels=self.dil_channels * factor,
                                                        dilations=self.dilations,
                                                        dropout=self.p_dropout,
                                                        activ=opt.dec_activ,
                                                        norm=opt.dec_norm,
                                                        padding=opt.dec_padding,
                                                        residual=False)]

        for i in range(self.num_upsamples):
            factor = int(factor / 2)
            self._layers = self._layers + [UpSampleConv(in_channels=self.dec_channels * prev_factor,
                                                        out_channels=self.dec_channels * factor,
                                                        dropout=self.p_dropout,
                                                        activ=opt.dec_activ,
                                                        norm=opt.dec_norm,
                                                        padding='zeros'
                                                        )]
            self._layers = self._layers + [MultiDilatedConv(in_channels=self.dec_channels * factor,
                                                            out_channels=self.dec_channels * factor,
                                                            dil_channels=self.dil_channels * factor,
                                                            dilations=self.dilations,
                                                            dropout=self.p_dropout,
                                                            activ=opt.dec_activ,
                                                            norm=opt.dec_norm,
                                                            padding=opt.dec_padding,
                                                            residual=False)]
            prev_factor = factor

        self._final_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=self.dec_channels * factor,
                out_channels=self.out_channels,
                kernel_size=7,
                padding=0,
                stride=1),
            self.activ_final
        ]

        self._layers = self._layers + self._final_layer

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class AggregatedLargeDilationDecoder(nn.Module):
    def __init__(self, opt, out_channels, type='continous'):
        super(AggregatedLargeDilationDecoder, self).__init__()
        self.out_channels = out_channels
        self.enc_channels = opt.enc_channels
        self.dec_channels = opt.dec_channels
        self.num_upsamples = opt.num_downsamples
        self.dil_channels = opt.dil_channels
        self.dilations = opt.dec_dilations
        final_dilation = self.dilations[-1]
        for i in range(1, self.num_upsamples + 1):
            self.dilations = self.dilations + [final_dilation + i * 2]
        # these are layer constructors, not implicit layers
        self.norm_layer = get_norm(opt.dec_norm)
        self.activ_layer = get_activ(opt.dec_activ)
        if type == 'continous':
            self.activ_final = nn.Tanh()
        if type == 'probabilities':
            self.activ_final = nn.Sigmoid()
        self.padding_layer = get_padding(opt.dec_padding)
        self.p_dropout = opt.dec_dropout
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []

        factor = 2 ** self.num_upsamples
        prev_factor = factor

        self._layers = self._layers + [MultiDilatedConv(in_channels=self.dec_channels * factor,
                                                        out_channels=self.dec_channels * factor,
                                                        dil_channels=self.dil_channels * factor,
                                                        dilations=self.dilations,
                                                        dropout=self.p_dropout * 2,
                                                        activ=opt.dec_activ,
                                                        norm=opt.dec_norm,
                                                        padding=opt.dec_padding,
                                                        residual=False)]

        for i in range(self.num_upsamples):
            factor = int(factor / 2)
            self._layers = self._layers + [UpSampleConv(in_channels=self.dec_channels * prev_factor,
                                                        out_channels=self.dec_channels * factor,
                                                        dropout=self.p_dropout,
                                                        activ=opt.dec_activ,
                                                        norm=opt.dec_norm,
                                                        padding='zeros'
                                                        )]

            self._layers = self._layers + [MultiDilatedConv(in_channels=self.dec_channels * factor,
                                                            out_channels=self.dec_channels * factor,
                                                            dil_channels=self.dil_channels * factor,
                                                            dilations=self.dilations[:-self.num_upsamples],
                                                            dropout=self.p_dropout,
                                                            activ=opt.dec_activ,
                                                            norm=opt.dec_norm,
                                                            padding=opt.dec_padding,
                                                            residual=False)]
            prev_factor = factor


        self._final_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=self.dec_channels * factor,
                out_channels=self.out_channels,
                kernel_size=7,
                stride=1),
            self.activ_final
        ]

        self._layers = self._layers + self._final_layer

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x
        #return self._layers(x)
