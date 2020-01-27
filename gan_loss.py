import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if self.mode == 'bce':
            # if probabilities are expected as output, run predictions through a Sigmoid
            prediction = F.sigmoid(prediction)
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
