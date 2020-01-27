from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torch
import numpy as np


def label_img_to_color(img):
    label_to_color = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        19: [81, 0, 81]
    }
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    # img_color = np.transpose(img_color, (2, 0, 1))
    return img_color


def color_segmentation(input):
    input = input.data.numpy()
    pred_label_imgs = np.argmax(input, axis=1)
    pred_label_imgs = pred_label_imgs.astype(np.uint8)[0]
    pred_label_color = torch.FloatTensor(label_img_to_color(pred_label_imgs))
    pred_label_color = (pred_label_color - 127.) / 128.
    return pred_label_color.permute(2, 0, 1).unsqueeze(0)


class CycleTriGanSummary(SummaryWriter):
    def __init__(self):
        super(CycleTriGanSummary, self).__init__()
        # self.device = 'cuda:0' + str(device)

    def write_scalars(self, prefix, scalar_dict, step):
        for name in scalar_dict:
            self.add_scalar(prefix + '/' + name, scalar_dict[name], step)
        self.flush()

    def write_visuals(self, visuals_dict, step):
        visuals = list(visuals_dict.items())
        visuals = [v[1].detach().cpu()[0].unsqueeze(0) for v in visuals]
        visuals[4] = color_segmentation(visuals[4])
        visuals[5] = color_segmentation(visuals[5])
        visuals[10] = color_segmentation(visuals[10])
        visuals[11] = color_segmentation(visuals[11])
        visuals = torch.stack(visuals, dim=1)[0]
        image_grid_A = vutils.make_grid(visuals[:6], padding=2, normalize=True)
        image_grid_B = vutils.make_grid(visuals[6:], padding=2, normalize=True)
        self.add_image('images_A_to_B', image_grid_A, step)
        self.add_image('images_B_to_A', image_grid_B, step)
        self.flush()
