import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from gan_loss import *
from training_utils import *
from basic_model import *
from basic_modules import *


class Discriminator(nn.Module):
    def __init__(self, opt, in_channels):
        super(Discriminator, self).__init__()
        self.dil_channels = opt.dil_channels
        self.dsc_channels = opt.dsc_channels
        self.dsc_layers = opt.dsc_layers
        self.dsc_scales = opt.dsc_scales
        self.in_channels = in_channels
        self.opt = opt
        self._build_layers()

    def _build_layers(self):
        self.heads = []
        self.modules = []
        self._init = [nn.Conv2d(in_channels=3,
                                out_channels=self.dsc_channels,
                                kernel_size=7,
                                stride=1,
                                padding=3)]
        for i in range(self.dsc_layers):
            self.modules = self.modules + [MultiDilatedConv(in_channels=self.dsc_channels * (2 ** i),
                                                            d_channels=self.dil_channels * (2 ** i),
                                                            out_channels=self.dsc_channels * (2 ** i),
                                                            dilations=[1, 3, 5])]
            self.modules = self.modules + [DownSampleConv(in_channels=self.dsc_channels * (2 ** i),
                                                          out_channels=self.dsc_channels * (2 ** (i + 1)))]

        self.modules = self._init + self.modules
        self.conv_block = nn.Sequential(*self.modules)

        size = int(self.opt.load_size // (2 ** self.dsc_layers)) + 1
        # possible scales = [1, 8, 16, 24, 32]
        for scale in self.dsc_scales:
            self.heads = self.heads + [nn.Conv2d(in_channels=self.dsc_channels * (2 ** self.dsc_layers),
                                                 out_channels=1,
                                                 kernel_size=size - scale,
                                                 padding=0,
                                                 stride=1)]
        self.heads = nn.ModuleList(self.heads)

    def forward(self, input):
        outputs = []
        x = self.conv_block(input)
        for scale_head in self.heads:
            outputs = outputs + [scale_head(x)]
        return outputs


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.low_channels = opt.low_channels
        self.dil_channels = opt.dil_channels
        self.num_downsamples = opt.num_downsamples
        self.dropout = opt.dropout
        self.num_stacks = opt.num_stack
        self.model = self._build_layers()

    def _build_layers(self):
        self.modules = []
        self._init = [nn.Conv2d(in_channels=3,
                                out_channels=self.low_channels,
                                kernel_size=7,
                                stride=1,
                                padding=3),
                      nn.InstanceNorm2d(self.low_channels),
                      nn.GELU()]
        for i in range(self.num_downsamples):
            self.modules = self.modules + [MultiDilatedConv(in_channels=self.low_channels * (2 ** i),
                                                            d_channels=self.dil_channels * (2 ** i),
                                                            out_channels=self.low_channels * (2 ** i),
                                                            dilations=[1, 2, 3],
                                                            dropout=self.dropout)]
            self.modules = self.modules + [DownSampleConv(in_channels=self.low_channels * (2 ** i),
                                                          out_channels=self.low_channels * (2 ** (i + 1)))]
        for i in range(self.num_stacks):
            self.modules = self.modules + [ResidualConv(num_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                        num_convs=3,
                                                        dropout=self.dropout)]
            self.modules = self.modules + [MultiDilatedConv(in_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                            d_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                            out_channels=self.dil_channels * (
                                                                    2 ** self.num_downsamples),
                                                            dilations=[1, 2, 3, 5, 7],
                                                            dropout=self.dropout)]
        self.modules = self._init + self.modules
        model = nn.Sequential(*self.modules)
        return model

    def forward(self, input):
        x = self.model(input)
        return x


class DownSamplingEncoder(nn.Module):
    def __init__(self, opt):
        super(DownSamplingEncoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.low_channels = opt.low_channels
        self.dil_channels = opt.dil_channels
        self.num_downsamples = opt.num_downsamples
        self.dropout = opt.dropout
        self.model = self._build_layers()

    def _build_layers(self):
        self.modules = []
        self._init = [nn.Conv2d(in_channels=3,
                                out_channels=self.low_channels,
                                kernel_size=7,
                                stride=1,
                                padding=3),
                      nn.InstanceNorm2d(self.low_channels),
                      nn.GELU()]
        for i in range(self.num_downsamples):
            self.modules = self.modules + [MultiDilatedConv(in_channels=self.low_channels * (2 ** i),
                                                            d_channels=self.dil_channels * (2 ** i),
                                                            out_channels=self.low_channels * (2 ** i),
                                                            dilations=[1, 2, 3],
                                                            dropout=self.dropout)]
            self.modules = self.modules + [DownSampleConv(in_channels=self.low_channels * (2 ** i),
                                                          out_channels=self.low_channels * (2 ** (i + 1)))]
        self.modules = self._init + self.modules
        model = nn.Sequential(*self.modules)
        return model

    def forward(self, input):
        x = self.model(input)
        return x


class MultiDilationStack(nn.Module):
    def __init__(self, opt):
        super(MultiDilationStack, self).__init__()
        self.enc_channels = opt.enc_channels
        self.low_channels = opt.low_channels
        self.dil_channels = opt.dil_channels
        self.num_downsamples = opt.num_downsamples
        self.dropout = opt.dropout
        self.num_stacks = opt.num_stack
        self.model = self._build_layers()

    def _build_layers(self):
        self.modules = []
        self._init = [nn.Conv2d(in_channels=3,
                                out_channels=self.low_channels,
                                kernel_size=7,
                                stride=1,
                                padding=3),
                      nn.InstanceNorm2d(self.low_channels),
                      nn.GELU()]

        for i in range(self.num_stacks):
            self.modules = self.modules + [ResidualConv(num_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                        num_convs=3,
                                                        dropout=self.dropout)]
            self.modules = self.modules + [MultiDilatedConv(in_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                            d_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                            out_channels=self.dil_channels * (
                                                                    2 ** self.num_downsamples),
                                                            dilations=[1, 2, 3, 5, 7],
                                                            dropout=self.dropout)]
        self.modules = self._init + self.modules
        model = nn.Sequential(*self.modules)
        return model

    def forward(self, input):
        x = self.model(input)
        return x


class UpsamplingDecoder(nn.Module):
    def __init__(self, opt):
        super(UpsamplingDecoder, self).__init__()
        self.dec_channels = opt.dec_channels
        self.dil_channels = opt.dil_channels
        self.num_downsamples = opt.num_downsamples
        self.num_upsamples = int(opt.num_downsamples // 2)
        self.droptpout = opt.dropout
        self.model = self._build_layers()

    def _build_layers(self):
        self.modules = []
        self.modules = self.modules + [UpSampleConv(in_channels=self.dil_channels * (2 ** self.num_downsamples),
                                                    out_channels=self.dec_channels * (2 ** self.num_downsamples))]
        for i in range(self.num_upsamples, 0, -1):
            self.modules = self.modules + [ResidualConv(num_channels=self.dec_channels * (2 ** i),
                                                        num_convs=3,
                                                        dropout=self.dropout)]
            self.modules = self.modules + [UpSampleConv(in_channels=self.dec_channels * (2 ** i),
                                                        out_channels=self.dec_channels * (2 ** (i - 1))
                                                        )]
        return nn.Sequential(*self.modules)

    def forward(self, input):
        return self.model(input)


class ColoringDecoder(nn.Module):
    def __init__(self, opt, out_channels):
        super(ColoringDecoder, self).__init__()
        self.dec_channels = opt.dec_channels
        self.dil_channels = opt.dil_channels
        self.out_channels = out_channels
        self.num_upsamples = int(opt.num_downsamples // 2)
        self.dropout = opt.dropout
        self.model = self._build_layers()

    def _build_layers(self):
        self.modules = []
        for i in range(self.num_upsamples, 0, -1):
            self.modules = self.modules + [ResidualConv(num_channels=self.dec_channels * (2 ** i),
                                                        num_convs=5,
                                                        dropout=self.dropout)]
            self.modules = self.modules + [UpSampleConv(in_channels=self.dec_channels * (2 ** i),
                                                        out_channels=self.dec_channels * (2 ** (i - 1))
                                                        )]
        self.modules = self.modules + [nn.Conv2d(in_channels=self.dec_channels * 2,
                                                 out_channels=self.dec_channels,
                                                 kernel_size=3,
                                                 padding=1, ),
                                       nn.InstanceNorm2d(self.dec_channels),
                                       nn.GELU()]

        self.modules = self.modules + [nn.ReflectionPad2d(3)]
        self.modules = self.modules + [nn.Conv2d(in_channels=self.dec_channels,
                                                 out_channels=self.out_channels,
                                                 kernel_size=7,
                                                 padding=0)]
        self.modules = self.modules + [nn.Tanh()]
        return nn.Sequential(*self.modules)


class Decoder(nn.Module):
    def __init__(self, opt, out_channels):
        super(Decoder, self).__init__()
        self.enc_channels = opt.enc_channels
        self.dil_channels = opt.dil_channels
        self.dec_channels = opt.dec_channels
        self.num_upsamples = opt.num_downsamples
        self.out_channels = out_channels
        self.dropout = opt.dropout
        self.model = self._build_layers()

    def _build_layers(self):
        self.modules = []
        self.modules = self.modules + [UpSampleConv(in_channels=self.dil_channels * (2 ** self.num_upsamples),
                                                    out_channels=self.dec_channels * (2 ** self.num_upsamples))]

        for i in range(self.num_upsamples, 1, -1):
            self.modules = self.modules + [ResidualConv(num_channels=self.dec_channels * (2 ** i),
                                                        num_convs=3,
                                                        dropout=self.dropout)]
            self.modules = self.modules + [UpSampleConv(in_channels=self.dec_channels * (2 ** i),
                                                        out_channels=self.dec_channels * (2 ** (i - 1))
                                                        )]
        self.modules = self.modules + [nn.Conv2d(in_channels=self.dec_channels * 2,
                                                 out_channels=self.dec_channels,
                                                 kernel_size=3,
                                                 padding=1, ),
                                       nn.InstanceNorm2d(self.dec_channels),
                                       nn.GELU()]

        self.modules = self.modules + [nn.ReflectionPad2d(3)]
        self.modules = self.modules + [nn.Conv2d(in_channels=self.dec_channels,
                                                 out_channels=self.out_channels,
                                                 kernel_size=7,
                                                 padding=0)]
        self.modules = self.modules + [nn.Tanh()]
        return nn.Sequential(*self.modules)

    def forward(self, input):
        return self.model(input)


class CycleTriGAN(BaseModel):
    def __init__(self, opt, segmentation_model):
        super(CycleTriGAN, self).__init__(opt)
        self.loss_names = [
            'discriminator_A',
            'decoder_rgb_A',
            'decoder_seg_A',
            'cycle_A',
            'identity_A',
            'discriminator_B',
            'decoder_rgb_B',
            'decoder_seg_B',
            'cycle_B',
            'identity_B',

        ]
        # self.opt = opt
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.lambda_idn = opt.lambda_idn
        self.lambda_aux = opt.lambda_aux
        self.trainable = opt.trainable

        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'identity_A', 'gt_seg_A', 'seg_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'identity_B', 'gt_seg_B', 'seg_B']
        self.visual_names = visual_names_A + visual_names_B

        self.norm = get_norm(opt.norm)
        if self.trainable:
            self.model_names = ['encoder_A',
                                'encoder_B',
                                'decoder_rgb_A',
                                'decoder_rgb_B',
                                'decoder_seg_A',
                                'decoder_seg_B',
                                'discriminator_A',
                                'discriminator_B']
        else:
            self.model_names = ['encoder_A',
                                'encoder_B',
                                'decoder_rgb_A',
                                'decoder_rgb_B',
                                'decoder_seg_A',
                                'decoder_seg_B']

        self.networks = []
        self.decoder_rgb_A = ColoringDecoder(self.opt, 3)
        self.decoder_rgb_B = ColoringDecoder(self.opt, 3)
        self.decoder_rgb_S = ColoringDecoder(self.opt, 20)
        self.decoder_shared = UpsamplingDecoder(self.opt)
        self.encoder_A = DownSamplingEncoder(self.opt)
        self.encoder_B = DownSamplingEncoder(self.opt)
        self.encoder_shared = MultiDilationStack(self.opt)
        self.networks = self.networks + [self.decoder_rgb_A, self.decoder_rgb_B, self.decoder_rgb_S, self.decoder_shared,
                                         self.encoder_A, self.encoder_B, self.encoder_shared]

        if self.trainable:
            self.discriminator_A = Discriminator(self.opt, 3)
            self.discriminator_B = Discriminator(self.opt, 3)
            self.networks = self.networks + [self.discriminator_A, self.discriminator_B]

        for i in range(len(self.networks)):
            self.networks[i] = init_model(self.networks[i], opt.init_type, opt.init_gain, opt.bias, self.gpu_ids)

        self.adversarial_objective = Objective(mode=opt.objective_type, target_real_label=opt.target_real_label,
                                               target_fake_label=opt.target_fake_label).to(self.device)
        self.cycle_objective = nn.MSELoss().to(self.device)
        self.identity_objective = nn.MSELoss().to(self.device)
        self.aux_objective = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer_generator = torch.optim.Adam(
            itertools.chain(
                self.encoder_A.parameters(),
                self.encoder_B.parameters(),
                self.encoder_shared.parameters(),
                self.decoder_shared.parameters(),
                self.decoder_rgb_S.parameters(),
                self.decoder_rgb_A.parameters(),
                self.decoder_rgb_B.parameters()
            ),
            lr=opt.lr,
            betas=(opt.beta_1, 0.999)
        )

        self.optimizer_discriminator = torch.optim.Adam(
            itertools.chain(
                self.discriminator_A.parameters(),
                self.discriminator_B.parameters(),
            ),
            lr=opt.lr,
            betas=(opt.beta_1, 0.999)
        )

        self.optimizers = [self.optimizer_generator, self.optimizer_discriminator]

        self.segmentation_model = segmentation_model.to(self.device)

    def set_input(self, input):
        self.real_A = input['rgb_A'].to(self.device)
        self.real_B = input['rgb_B'].to(self.device)
        self.gt_seg_A = self.segmentation_model(self.real_A)
        self.gt_seg_B = self.segmentation_model(self.real_B)

    def forward(self):
        self.e_A = self.encoder_shared(self.encoder_A(self.real_A))
        self.e_B = self.encoder_shared(self.encoder_B(self.real_B))
        self.d_A = self.decoder_shared(self.e_A)
        self.d_B = self.decoder_shared(self.e_B)
        self.fake_A = self.decoder_rgb_A(self.d_B)
        self.fake_B = self.decoder_rgb_B(self.d_A)
        self.seg_A = self.decoder_rgb_S(self.d_A)
        self.seg_B = self.decoder_rgb_S(self.d_B)

        self.rec_e_A = self.encoder_shared(self.encoder_B(self.fake_B))
        self.rec_A = self.decoder_rgb_A(self.decoder_shared(self.rec_e_A))
        self.rec_e_B = self.encoder_shared(self.encoder_A(self.fake_A))
        self.rec_B = self.decoder_rgb_B(self.decoder_shared(self.rec_e_B))

    def backward_D(self, discriminator, real, fake):
        pred_real = discriminator(real)
        pred_fake = discriminator(fake.detach())
        loss_real = 0.
        loss_fake = 0.
        for (pr, pf) in zip(pred_real, pred_fake):
            loss_real += self.adversarial_objective(pr, True)
            loss_fake += self.adversarial_objective(pf, False)
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_discriminator_A(self):
        fake_A = self.fake_A
        self.loss_discriminator_A = self.backward_D(self.discriminator_A, self.real_A, fake_A)

    def backward_discriminator_B(self):
        fake_B = self.fake_B
        self.loss_discriminator_B = self.backward_D(self.discriminator_B, self.real_B, fake_B)

    def backward_generator(self):
        if self.lambda_idn > 0:
            self.identity_A = self.decoder_rgb_A(self.d_A)
            self.identity_B = self.decoder_rgb_B(self.d_B)
            self.loss_identity_A = self.identity_objective(self.identity_A,
                                                           self.real_A) * self.lambda_A * self.lambda_idn
            self.loss_identity_B = self.identity_objective(self.identity_B,
                                                           self.real_B) * self.lambda_B * self.lambda_idn
        else:
            self.loss_identity_A = 0.
            self.loss_identity_B = 0.
        self.loss_decoder_rgb_A = 0.
        self.loss_decoder_rgb_B = 0.
        self.pred_discriminator_B = self.discriminator_B(self.fake_B)
        self.pred_discriminator_A = self.discriminator_A(self.fake_B)
        for pred_A, pred_B in zip(self.pred_discriminator_A, self.pred_discriminator_B):
            self.loss_decoder_rgb_B += self.adversarial_objective(pred_B, True)
            self.loss_decoder_rgb_A += self.adversarial_objective(pred_A, True)
        self.loss_cycle_A = self.cycle_objective(self.rec_A, self.real_A) * self.lambda_A
        self.loss_cycle_B = self.cycle_objective(self.rec_B, self.real_B) * self.lambda_B
        self.loss_decoder_seg_A = self.aux_objective(F.softmax(self.seg_A), self.gt_seg_A) * self.lambda_aux
        self.loss_decoder_seg_B = self.aux_objective(F.softmax(self.seg_B), self.gt_seg_B) * self.lambda_aux
        self.loss = self.loss_decoder_rgb_A + self.loss_decoder_rgb_B + \
                    self.loss_decoder_seg_A + self.loss_decoder_seg_B + \
                    self.loss_cycle_A + self.loss_cycle_B + \
                    self.loss_identity_A + self.loss_identity_B
        self.loss.backward()

    def optimize(self):
        self.forward()
        self.set_requires_grad([self.discriminator_A, self.discriminator_B], False)
        self.optimizer_generator.zero_grad()
        self.backward_generator()
        self.optimizer_generator.step()

        self.set_requires_grad([self.discriminator_A, self.discriminator_B], True)
        self.optimizer_discriminator.zero_grad()
        self.backward_discriminator_A()
        self.backward_discriminator_B()
        self.optimizer_discriminator.step()
