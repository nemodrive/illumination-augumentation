import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from gan_loss import *
from training_utils import *
from basic_model import *
from basic_modules import *
from torch.nn.utils import clip_grad_norm_
import importlib


def get_encoder_model(model_name):
    model_file = 'encoders'
    model_lib = importlib.import_module(model_file)
    model = None
    model_class = model_name
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_class.lower() and issubclass(cls, nn.Module):
            model = cls
    return model


def get_decoder_model(model_name):
    model_file = 'decoders'
    model_lib = importlib.import_module(model_file)
    model = None
    model_class = model_name
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_class.lower() and issubclass(cls, nn.Module):
            model = cls
    return model


def get_discriminator_model(model_name):
    model_file = 'discriminators'
    model_lib = importlib.import_module(model_file)
    model = None
    model_class = model_name
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_class.lower() and issubclass(cls, nn.Module):
            model = cls
    return model


class CycleTriGANSharedSegmentationDecoder(BaseModel):
    def __init__(self, opt, segmentation_model):
        super(CycleTriGANSharedSegmentationDecoder, self).__init__(opt)
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

        self.gradient_norm_names = [
            'decoder_rgb_A',
            'decoder_rgb_B',
            'decoder_seg',
            'discriminator_A',
            'discriminator_B',
            'encoder_A',
            'encoder_B'
        ]

        self.opt = opt
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.lambda_idn = opt.lambda_idn
        self.lambda_aux = opt.lambda_aux
        self.lambda_aux_A = opt.lambda_aux_A
        self.lambda_aux_B = opt.lambda_aux_B
        self.trainable = opt.trainable

        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'identity_A', 'gt_seg_A', 'seg_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'identity_B', 'gt_seg_B', 'seg_B']
        self.visual_names = visual_names_A + visual_names_B

        if self.trainable:
            self.model_names = ['encoder_A',
                                'encoder_B',
                                'decoder_rgb_A',
                                'decoder_rgb_B',
                                'decoder_seg',
                                'discriminator_A',
                                'discriminator_B']
        else:
            self.model_names = ['encoder_A',
                                'encoder_B',
                                'decoder_rgb_A',
                                'decoder_rgb_B',
                                'decoder_seg']

        self.networks = []
        decoder_model = get_decoder_model(opt.dec_model)
        encoder_model = get_encoder_model(opt.enc_model)
        discriminator_model = get_discriminator_model(opt.dsc_model)

        self.encoder_A = encoder_model(opt)
        self.encoder_B = encoder_model(opt)

        latent_size = self.encoder_A.get_output_channels()

        segmentation_decoder = decoder_model(latent_size, 20, opt, type='probabilities')
        self.decoder_rgb_A = decoder_model(latent_size, 3, opt, type='continous')
        self.decoder_rgb_B = decoder_model(latent_size, 3, opt, type='continous')
        self.decoder_seg = segmentation_decoder

        self.networks = self.networks + [self.decoder_rgb_A, self.decoder_rgb_B, self.decoder_seg,
                                         self.encoder_A, self.encoder_B]

        # refference for gradient clipping
        self.generative_networks = self.networks

        self.fake_pool_A = DiscriminativePool(opt)
        self.fake_pool_B = DiscriminativePool(opt)

        if self.trainable:
            self.discriminator_A = discriminator_model(opt)
            self.discriminator_B = discriminator_model(opt)
            self.networks = self.networks + [self.discriminator_A, self.discriminator_B]

        for i in range(len(self.networks)):
            self.networks[i] = init_model(self.networks[i], opt.init_type, opt.init_gain, opt.bias, self.gpu_ids)

        # label smoothing for adversarial-generative tasks
        self.generative_objective = AdversarialObjective(mode=opt.adversarial_objective,
                                                         target_real_label=1.0,
                                                         target_fake_label=0.0).to(self.device)
        self.discriminative_objective = AdversarialObjective(mode=opt.adversarial_objective,
                                                             target_real_label=0.9,
                                                             target_fake_label=0.0, ).to(self.device)
        self.cycle_objective = ReconstructionObjective(opt.reconstruction_objective).to(self.device)
        self.identity_objective = ReconstructionObjective(opt.reconstruction_objective).to(self.device)
        self.aux_objective = nn.BCELoss().to(self.device)

        self.optimizer_generator = torch.optim.Adam(
            itertools.chain(
                self.encoder_A.parameters(),
                self.encoder_B.parameters(),
                self.decoder_seg.parameters(),
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
        self.gt_seg_A = input['seg_A'].to(self.device)
        self.gt_seg_B = input['seg_B'].to(self.device)

        self.gt_seg_A = self.segmentation_model(self.gt_seg_A).detach()
        self.gt_seg_B = self.segmentation_model(self.gt_seg_B).detach()

        self.gt_seg_A = F.upsample(self.gt_seg_A, self.opt.load_size)
        self.gt_seg_B = F.upsample(self.gt_seg_B, self.opt.load_size)

        self.gt_seg_A = F.sigmoid(self.gt_seg_A)
        self.gt_seg_B = F.sigmoid(self.gt_seg_B)

    def forward(self):
        self.e_A = self.encoder_A(self.real_A)
        self.e_B = self.encoder_B(self.real_B)
        self.fake_A = self.decoder_rgb_A(self.e_B)
        self.fake_B = self.decoder_rgb_B(self.e_A)

        self.fake_pool_A.add_to_pool(self.fake_A)
        self.fake_pool_B.add_to_pool(self.fake_B)

        self.seg_A = self.decoder_seg(self.e_A)
        self.seg_B = self.decoder_seg(self.e_B)
        self.rec_A = self.decoder_rgb_A(self.encoder_B(self.fake_B))
        self.rec_B = self.decoder_rgb_B(self.encoder_A(self.fake_A))

    def backward_D(self, discriminator, real, fake):
        pred_real = discriminator(real)
        pred_fake = discriminator(fake.detach())
        loss_real = 0.
        loss_fake = 0.
        for (pr, pf) in zip(pred_real, pred_fake):
            loss_real += self.discriminative_objective(pr, True)
            loss_fake += self.discriminative_objective(pf, False)
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D = loss_D / len(pred_real)
        loss_D.backward()
        return loss_D

    def backward_discriminator_A(self):
        fake_A = self.fake_pool_A.fetch_candidates()
        self.loss_discriminator_A = self.backward_D(self.discriminator_A, self.real_A, fake_A)

    def backward_discriminator_B(self):
        fake_B = self.fake_pool_B.fetch_candidates()
        self.loss_discriminator_B = self.backward_D(self.discriminator_B, self.real_B, fake_B)

    def backward_generator(self):
        if self.lambda_idn > 0:
            self.identity_A = self.decoder_rgb_A(self.encoder_A(self.real_A))
            self.identity_B = self.decoder_rgb_B(self.encoder_B(self.real_B))
            self.loss_identity_A = self.identity_objective(self.identity_A,
                                                           self.real_A) * self.lambda_idn
            self.loss_identity_B = self.identity_objective(self.identity_B,
                                                           self.real_B) * self.lambda_idn
        else:
            self.loss_identity_A = 0.
            self.loss_identity_B = 0.
        self.loss_decoder_rgb_A = 0.
        self.loss_decoder_rgb_B = 0.
        self.pred_discriminator_A = self.discriminator_A(self.fake_A)
        self.pred_discriminator_B = self.discriminator_B(self.fake_B)
        for pred_A, pred_B in zip(self.pred_discriminator_A, self.pred_discriminator_B):
            self.loss_decoder_rgb_A += self.generative_objective(pred_A, True)
            self.loss_decoder_rgb_B += self.generative_objective(pred_B, True)
        self.loss_decoder_rgb_A = self.loss_decoder_rgb_A / len(self.opt.dsc_scales)
        self.loss_decoder_rgb_B = self.loss_decoder_rgb_B / len(self.opt.dsc_scales)
        self.loss_cycle_A = self.cycle_objective(self.rec_A, self.real_A) * self.lambda_A
        self.loss_cycle_B = self.cycle_objective(self.rec_B, self.real_B) * self.lambda_B
        self.loss_decoder_seg_A = self.aux_objective(self.seg_A, self.gt_seg_A) * self.lambda_aux * self.lambda_aux_A
        self.loss_decoder_seg_B = self.aux_objective(self.seg_B, self.gt_seg_B) * self.lambda_aux * self.lambda_aux_B
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

        # for network in self.generative_networks:
        #    gnorm_network = clip_grad_norm_(network.parameters(), self.opt.gen_max_gnorm)

        # needs to be refactored
        self.gnorm_decoder_rgb_A = clip_grad_norm_(self.decoder_rgb_A.parameters(), self.opt.gen_max_gnorm)
        self.gnorm_decoder_rgb_B = clip_grad_norm_(self.decoder_rgb_B.parameters(), self.opt.gen_max_gnorm)
        self.gnorm_decoder_seg = clip_grad_norm_(self.decoder_seg.parameters(), self.opt.gen_max_gnorm)
        self.gnorm_encoder_A = clip_grad_norm_(self.encoder_A.parameters(), self.opt.gen_max_gnorm)
        self.gnorm_encoder_B = clip_grad_norm_(self.encoder_B.parameters(), self.opt.gen_max_gnorm)

        self.optimizer_generator.step()

        self.set_requires_grad([self.discriminator_A, self.discriminator_B], True)
        self.optimizer_discriminator.zero_grad()
        self.backward_discriminator_A()
        self.backward_discriminator_B()

        # needs to be refactored
        self.gnorm_discriminator_A = clip_grad_norm_(self.discriminator_A.parameters(), self.opt.dsc_max_gnorm)
        self.gnorm_discriminator_B = clip_grad_norm_(self.discriminator_B.parameters(), self.opt.dsc_max_gnorm)
        self.optimizer_discriminator.step()
