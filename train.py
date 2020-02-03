import time
import importlib
import yaml
import torch
from deeplabv3 import DeepLabV3
from argparse import Namespace
from basic_model import BaseModel
from dataset import UnpairedDataset, SegmentationUnpairedDataset
from visualizer import CycleTriGanSummary
import shutil

segment_network = DeepLabV3("eval_val",
                            project_dir="drive_augumentation").cuda()
segment_network.load_state_dict(torch.load("pretrained_models/model_13_2_2_2_epoch_580.pth"))
segment_network.eval()


def get_model(model_name):
    model_file = model_name
    model_lib = importlib.import_module(model_file)
    model = None
    model_class = model_name.replace('_', '')
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_class.lower() and issubclass(cls, BaseModel):
            model = cls

    return model


def create_model(opt, segment_network):
    model = get_model(opt.model)
    instance = model(opt, segment_network)
    print("Model [%s] was created" % type(instance).__name__)
    return instance


if __name__ == '__main__':
    with open('config.yml', 'r') as cfg_file:
        opt = yaml.load(cfg_file)
        opt = Namespace(**opt)
    model = create_model(opt, segment_network)
    dataset = SegmentationUnpairedDataset(opt.root_path, opt.load_size, opt.seg_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=8)
    model.setup(opt)

    config_copy = shutil.copy('config.yml', model.save_dir + '/config.yml')
    visualizer = CycleTriGanSummary()

    total_iters = 0

    for epoch in range(1, opt.epoch_count + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iters = 0

        for idx, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                time_data = iter_start_time - iter_data_time
            total_iters = total_iters + opt.batch_size
            epoch_iters = epoch_iters + opt.batch_size
            model.set_input(data)
            model.optimize()

            if total_iters % opt.display_freq == 0:
                model.compute_visuals()
                visualizer.write_visuals(visuals_dict=model.get_current_visuals(), step=total_iters)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                gnorms = model.get_current_gradient_norms()
                time_compute = (time.time() - iter_start_time) / opt.batch_size
                visualizer.write_scalars('Loss', losses, total_iters)
                visualizer.write_scalars('Gradient Norms', gnorms, total_iters)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, total_iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()
