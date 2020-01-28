import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class AdversarialObjective(nn.Module):
    def __init__(self, mode, target_real_label=1.0, target_fake_label=0.0):
        super(AdversarialObjective, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.mode = mode

        if mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'mse':
            self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class ReconstructionObjective(nn.Module):
    def __init__(self, mode):
        super(ReconstructionObjective, self).__init__()
        if mode == 'L1':
            self.loss = nn.L1Loss()
        elif mode == 'L2':
            self.loss = nn.MSELoss()

    def __call__(self, prediction, target):
        return self.loss(prediction, target)


class DiscriminativePool():
    def __init__(self, opt):
        self.images = []
        self.max_size = opt.pool_max_size
        self.add_prob = opt.pool_add_prob
        self.size = 0

    def add_to_pool(self, images):
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if our buffer has not reached max capacity yet
            if self.size < self.max_size:
                self.size = self.size + 1
                self.images = self.images + [image]
            else:
                p = random.uniform(0, 1)
                # if rolled probability is high enough an old image is swapped with
                # a freshly generated one
                if p > 1 - self.add_prob:
                    swap_idx = random.randint(0, self.max_size - 1)
                    self.images[swap_idx] = image

    def fetch_candidates(self):
        # create batch of images
        test_images = torch.cat(self.images, 0)
        return test_images
