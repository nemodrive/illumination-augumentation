import torch
import torch.nn
from training_utils import get_norm, get_padding, get_activ
from basic_modules import *


class BaseEncoder(nn.Module):
    def __init__(self, opt):
        super(BaseEncoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.num_downsamples = opt.num_downsamples
        self.padding_layer = get_padding(opt.enc_padding)
        self.activ_layer = get_activ(opt.enc_activ)
        self.norm_layer = get_norm(opt.enc_norm)
        self.p_dropout = opt.enc_dropout
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []
        self._init_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=3,
                out_channels=self.enc_channels,
                kernel_size=7,
                stride=1)
        ]

        self._layers = self._layers + self._init_layer

        factor = 1
        prev_factor = 1
        for i in range(self.num_downsamples):
            factor = factor * 2
            self._layers = self._layers + [DownSampleConv(in_channels=self.enc_channels * prev_factor,
                                                          out_channels=self.enc_channels * factor,
                                                          dropout=self.p_dropout,
                                                          activ=opt.enc_activ,
                                                          norm=opt.enc_norm,
                                                          padding=opt.enc_padding
                                                          )]
            prev_factor = factor

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class BaseDilationEncoder(nn.Module):
    def __init__(self, opt):
        super(BaseDilationEncoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.num_downsamples = opt.num_downsamples
        self.dilations = opt.enc_dilations
        self.dil_channels = opt.dil_channels
        self.padding_layer = get_padding(opt.enc_padding)()
        self.activ_layer = get_activ(opt.enc_activ)
        self.norm_layer = get_norm(opt.enc_norm)
        self.p_dropout = opt.enc_dropout
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []
        self._init_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=3,
                out_channels=self.enc_channels,
                kernel_size=7,
                stride=1)
        ]

        self._layers = self._layers + self._init_layer
        factor = 1
        prev_factor = 1
        for i in range(self.num_downsamples):
            factor = factor * 2
            self._layers = self._layers + [MultiDilatedConv(in_channels=self.enc_channels * prev_factor,
                                                            out_channels=self.enc_channels * prev_factor,
                                                            dil_channels=self.dil_channels * prev_factor,
                                                            dilations=self.dilations,
                                                            dropout=self.p_dropout,
                                                            activ=opt.enc_activ,
                                                            norm=opt.enc_norm,
                                                            padding=opt.enc_padding,
                                                            residual=False)]

            self._layers = self._layers + [DownSampleConv(in_channels=self.enc_channels * prev_factor,
                                                          out_channels=self.enc_channels * factor,
                                                          dropout=self.p_dropout,
                                                          activ=opt.enc_activ,
                                                          norm=opt.enc_norm,
                                                          padding=opt.enc_padding
                                                          )]
            prev_factor = factor

        self._layers = nn.Sequential(*self._layers)

    def forward(self, input):
        # x = input
        # for layer in self._layers:
        #    x = layer(x)
        x = self._layers(input)
        return x


class AggregatedLargeDilationEncoder(nn.Module):
    def __init__(self, opt):
        super(AggregatedLargeDilationEncoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.num_downsamples = opt.num_downsamples
        self.dilations = opt.enc_dilations
        final_dilation = self.dilations[-1]
        for i in range(1, self.num_downsamples + 1):
            # for each downsample add an increasing dilation
            # why? no reason
            self.dilations = self.dilations + [final_dilation + i * 2]
        self.dil_channels = opt.dil_channels
        self.padding_layer = get_padding(opt.enc_padding)
        self.activ_layer = get_activ(opt.enc_activ)
        self.norm_layer = get_norm(opt.enc_norm)
        self.p_dropout = opt.enc_dropout
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []
        self._init_layer = [
            self.padding_layer(3),
            nn.Conv2d(
                in_channels=3,
                out_channels=self.enc_channels,
                kernel_size=7,
                stride=1)
        ]

        self._layers = self._layers + self._init_layer
        factor = 1
        prev_factor = 1
        for i in range(self.num_downsamples):
            factor = factor * 2
            # dilating before downsampling
            self._layers = self._layers + [MultiDilatedConv(in_channels=self.enc_channels * prev_factor,
                                                            out_channels=self.enc_channels * prev_factor,
                                                            dil_channels=self.dil_channels * prev_factor,
                                                            dilations=self.dilations[:-self.num_downsamples],
                                                            dropout=self.p_dropout,
                                                            activ=opt.enc_activ,
                                                            norm=opt.enc_norm,
                                                            padding=opt.enc_padding,
                                                            residual=False)]

            self._layers = self._layers + [DownSampleConv(in_channels=self.enc_channels * prev_factor,
                                                          out_channels=self.enc_channels * factor,
                                                          dropout=self.p_dropout,
                                                          activ=opt.enc_activ,
                                                          norm=opt.enc_norm,
                                                          padding=opt.enc_padding
                                                          )]
            prev_factor = factor

        # increase dropout factor due to increased number of dilations used
        # using residual connections. why? no reason
        self._layers = self._layers + [MultiDilatedConv(in_channels=self.enc_channels * factor,
                                                        out_channels=self.enc_channels * factor,
                                                        dil_channels=self.dil_channels * factor,
                                                        dilations=self.dilations,
                                                        dropout=self.p_dropout * 2,
                                                        activ=opt.enc_activ,
                                                        norm=opt.enc_norm,
                                                        padding=opt.enc_padding,
                                                        residual=True)]
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        # x = self._layers(input)
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class ScalingResidualEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super(ScalingResidualEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_layer = get_padding(opt.latent_padding)
        self.activ_layer = get_activ(opt.latent_activ)
        self.norm_layer = get_norm(opt.latent_norm)
        self.p_dropout = opt.latent_dropout
        self._build_layers(opt)

    def _build_layers(self, opt):
        self._layers = []
        if self.in_channels < self.out_channels:
            self.conv_layer = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        )
            self._layers = self._layers + [
                self.padding_layer(1),
                self.conv_layer,
                self.norm_layer(self.out_channels),
                self.activ_layer(),
            ]
        self._layers = self._layers + [ResidualConv(num_channels=self.out_channels,
                                                    num_blocks=opt.latent_blocks,
                                                    dropout=opt.latent_dropout,
                                                    activ=opt.latent_activ,
                                                    norm=opt.latent_norm,
                                                    padding=opt.laten_padding)
                                       ]
        if self.in_channels > self.out_channels:
            self.conv_layer = nn.Conv2d(in_channels=self.out_channels,
                                        out_channels=self.in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        )
            self._layers = self._layers + [
                self.padding_layer(1),
                self.conv_layer,
                self.norm_layer(self.out_channels),
                self.activ_layer(),
            ]
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        # x = self._layers(input)
        x = input
        for layer in self._layers:
            x = layer(x)
        return x
