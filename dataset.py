import torch
import torchvision.transforms.functional as F
import torch.utils.data
from random import randint
import PIL.Image as Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', 'JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def base_loader(path, size):
    img = Image.open(path).convert('RGB')
    img = F.resize(img, size)
    img = F.to_tensor(img)
    img = F.normalize(img, (0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,), inplace=True)
    return img


def resn_loader(path, size):
    img = Image.open(path).convert('RGB')
    img = F.resize(img, size)
    img = F.to_tensor(img)
    img = F.normalize(img, (0.485, 0.456, 0.406,), (0.229, 0.224, 0.225,), inplace=True)
    return img

class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, load_size):
        super(UnpairedDataset, self).__init__()
        self.paths_A = make_dataset(root_path + '/trainA')
        self.paths_B = make_dataset(root_path + '/trainB')
        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)
        self.img_h = load_size
        self.img_w = load_size

    def __getitem__(self, index):
        path_A = self.paths_A[index % self.size_A]
        path_B = self.paths_B[randint(0, self.size_B - 1)]

        img_A = base_loader(path_A, (self.img_w, self.img_h))
        img_B = base_loader(path_B, (self.img_w, self.img_h))

        example = {}
        example['rgb_A'] = img_A
        example['rgb_B'] = img_B
        example['pth_A'] = path_A
        example['pth_B'] = path_B

        return example

    def __len__(self):
        return max(self.size_B, self.size_A)


class SegmentationUnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, load_size, seg_size):
        super(SegmentationUnpairedDataset, self).__init__()
        self.paths_A = make_dataset(root_path + '/trainA')
        self.paths_B = make_dataset(root_path + '/trainB')
        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)
        self.img_h = load_size
        self.img_w = load_size
        self.seg_h = seg_size
        self.seg_w = seg_size

    def __getitem__(self, index):
        path_A = self.paths_A[index % self.size_A]
        path_B = self.paths_B[randint(0, self.size_B - 1)]

        img_A = base_loader(path_A, (self.img_w, self.img_h))
        img_B = base_loader(path_B, (self.img_w, self.img_h))
        seg_A = resn_loader(path_A, (self.seg_w, self.seg_h))
        seg_B = resn_loader(path_B, (self.seg_w, self.seg_h))

        example = {}
        example['rgb_A'] = img_A
        example['rgb_B'] = img_B
        example['seg_A'] = seg_A
        example['seg_B'] = seg_B
        example['pth_A'] = path_A
        example['pth_B'] = path_B

        return example

    def __len__(self):
        return max(self.size_B, self.size_A)