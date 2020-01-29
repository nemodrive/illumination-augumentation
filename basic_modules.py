import torch
import torch.nn as nn
import torch.nn.functional as F
from training_utils import get_norm, get_activ, get_padding


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 1024, 1, 1)


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., activ='gelu', norm='instance', padding='reflection'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = get_norm(norm)
        self.activ_layer = get_activ(activ)
        self.padding_layer = get_padding(padding)
        self.padding_mode = padding
        self.p_dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self._layers = []
        self.upsample = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1,
                                           padding_mode=self.padding_mode)

        self._layers = self._layers + [
            self.upsample,
            self.norm_layer(self.out_channels),
            self.activ_layer(),
            nn.Dropout2d(self.p_dropout)
        ]

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0., activ='gelu', norm='instance', padding='reflection'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = get_norm(norm)
        self.activ_layer = get_activ(activ)
        self.padding_layer = get_padding(padding)
        self.p_dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self._layers = []
        self.downsample = nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    )

        self._layers = self._layers + [
            self.padding_layer(1),
            self.downsample,
            self.norm_layer(self.out_channels),
            self.activ_layer(),
            nn.Dropout2d(self.p_dropout)
        ]

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class SquashConv(nn.Module):
    def __init__(self, in_channels, out_channels, activ='gelu', norm='instance'):
        super().__init__()
        self.in_channels = in_channels
        self._out_channels = out_channels
        self.activ_layer = get_activ(activ)
        self.norm_layer = get_norm(type)
        self._build_layers()

    def _build_layers(self):
        self._layers = [
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=1,
                      padding=0),
            self.norm_layer(self.out_channels),
            self.activ_layer()
        ]
        self._layers = nn.ModuleList(self._layer)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class ResidualConv(nn.Module):
    def __init__(self, num_channels, num_layers, dropout=0., activ='gelu', norm='instance', padding='reflection'):
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.norm_layer = get_norm(norm)
        self.activ_layer = get_activ(activ)
        self.padding_layer = get_padding(padding)
        self.p_dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self._layers = []
        for _ in range(self.num_layers):
            self.conv_layer = nn.Conv2d(in_channels=self.num_channels,
                                        out_channels=self.num_channels,
                                        kernel_size=3,
                                        stride=1,
                                        )

            self._layers = self._layers + [
                self.padding_layer(1),
                self.conv_layer,
                self.norm_layer(self.num_channels),
                self.activ_layer(),
            ]

        self._layers = self._layers + [nn.Dropout2d(self.p_dropout)]
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers[:-1]:
            x = layer(x)
        x = x + input
        x = self._layers[-1](x)
        return x


class ChainedResidualConv(nn.Module):
    def __init__(self, num_channels, num_blocks=1, num_layers=3, dropout=0., activ='gelu', norm='instance',
                 padding='reflection'):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.norm_layer = norm
        self.activ_layer = activ
        self.padding_layer = padding
        self.p_dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self._layers = []
        for _ in range(self.num_blocks):
            self._layers = self._layers + [ResidualConv(num_channels=self.num_channels,
                                                        num_layers=self.num_layers,
                                                        dropout=self.p_dropout,
                                                        activ=self.activ_layer,
                                                        norm=self.norm_layer,
                                                        padding=self.padding_layer
                                                        )]
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class ScalingResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, num_layers, opt):
        super(ScalingResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_channels = max(self.in_channels, self.out_channels)
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

        self._layers = self._layers + [ChainedResidualConv(num_channels=self.num_channels,
                                                           num_blocks=self.num_blocks,
                                                           num_layers=self.num_layers,
                                                           dropout=self.p_dropout,
                                                           activ=opt.latent_activ,
                                                           norm=opt.latent_norm,
                                                           padding=opt.latent_padding)]

        if self.in_channels > self.out_channels:
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
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        # x = self._layers(input)
        x = input
        for layer in self._layers:
            x = layer(x)
        return x


class MultiDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dil_channels, dilations, dropout=0., activ='gelu', norm='instance',
                 padding='reflection', residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil_channels = dil_channels
        self.dilations = dilations
        self.num_dilations = len(dilations)
        self.residual = residual
        self.norm_layer = get_norm(norm)
        self.activ_layer = get_activ(activ)
        self.padding_layer = get_padding(padding)
        self.p_dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self._layers = []
        self._dilation_layers = []
        for dilation in self.dilations:
            dilation_layer = [
                self.padding_layer(dilation),
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.dil_channels,
                          kernel_size=3,
                          dilation=dilation),
                self.norm_layer(self.dil_channels),
                self.activ_layer(),
                nn.Dropout2d(self.p_dropout)
            ]
            self._dilation_layers = self._dilation_layers + [dilation_layer]

        self._squash_layer = [
            nn.Conv2d(in_channels=self.dil_channels * self.num_dilations,
                      out_channels=self.out_channels,
                      kernel_size=1,
                      padding=0),
            self.norm_layer(self.out_channels),
            self.activ_layer()
        ]

        for layer in self._dilation_layers:
            self._layers = self._layers + layer
        self._layers = self._layers + self._squash_layer

        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        outputs = []
        for dilation_layer in self._dilation_layers:
            x = input
            for layer in dilation_layer:
                x = layer(x)
            outputs = outputs + [x]

        x = torch.cat(outputs, dim=1)

        for layer in self._squash_layer:
            x = layer(x)

        if self.residual:
            x = x + input

        return x
