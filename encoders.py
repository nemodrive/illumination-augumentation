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

        self.output_size = self.enc_channels * factor
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x

    def get_output_channels(self):
        return self.output_size


class BaseDilationEncoder(nn.Module):
    def __init__(self, opt):
        super(BaseDilationEncoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.num_downsamples = opt.num_downsamples
        self.dilations = opt.enc_dilations
        self.num_dilations = len(self.dilations)
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
            '''
            self._layers = self._layers + [SquashMultiDilatedConv(in_channels=self.enc_channels * prev_factor,
                                                                  out_channels=self.enc_channels * prev_factor,
                                                                  dil_channels=self.dil_channels * prev_factor,
                                                                  dilations=self.dilations,
                                                                  dropout=self.p_dropout,
                                                                  activ=opt.enc_activ,
                                                                  norm=opt.enc_norm,
                                                                  padding=opt.enc_padding,
                                                                  residual=False)]
            '''
            self._layers = self._layers + [
                MultiDilatedConv(in_channels=self.enc_channels * prev_factor,
                                 dil_channels=self.dil_channels * prev_factor,
                                 dilations=self.dilations,
                                 dropout=self.p_dropout,
                                 activ=opt.enc_activ,
                                 padding=opt.enc_padding,
                                 )]
            self._layers = self._layers + [
                DownSampleConv(in_channels=self.dil_channels * self.num_dilations * prev_factor,
                               out_channels=self.enc_channels * factor,
                               dropout=self.p_dropout,
                               activ=opt.enc_activ,
                               norm=opt.enc_norm,
                               padding=opt.enc_padding
                               )]
            prev_factor = factor

        self.output_size = self.enc_channels * factor
        self._layers = nn.Sequential(*self._layers)

    def forward(self, input):
        # x = input
        # for layer in self._layers:
        #    x = layer(x)
        x = self._layers(input)
        return x

    def get_output_channels(self):
        return self.output_size


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
            self._layers = self._layers + [SquashMultiDilatedConv(in_channels=self.enc_channels * prev_factor,
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
        self._layers = self._layers + [SquashMultiDilatedConv(in_channels=self.enc_channels * factor,
                                                              out_channels=self.enc_channels * factor,
                                                              dil_channels=self.dil_channels * factor,
                                                              dilations=self.dilations,
                                                              dropout=self.p_dropout * 2,
                                                              activ=opt.enc_activ,
                                                              norm=opt.enc_norm,
                                                              padding=opt.enc_padding,
                                                              residual=True)]
        self._layers = nn.ModuleList(self._layers)
        self.output_size = self.enc_channels * factor

    def forward(self, input):
        # x = self._layers(input)
        x = input
        for layer in self._layers:
            x = layer(x)
        return x

    def get_output_channels(self):
        return self.output_size


class DilationIntoResidualsEncoder(nn.Module):
    def __init__(self, opt):
        super(DilationIntoResidualsEncoder, self).__init__()
        self.opt = opt
        self._build_layers()

    def _build_layers(self):
        self._layers = []
        dilation_model = BaseDilationEncoder(self.opt)
        dilation_model_out_channels = dilation_model.get_output_channels()
        residual_model = ScalingResidualBlock(in_channels=dilation_model_out_channels,
                                              out_channels=dilation_model_out_channels * 2,
                                              num_blocks=self.opt.latent_blocks,
                                              num_layers=self.opt.latent_layers,
                                              opt=self.opt)
        self.output_size = dilation_model_out_channels * 2
        self._layers = self._layers + [dilation_model, residual_model]
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x

    def get_output_channels(self):
        return self.output_size
