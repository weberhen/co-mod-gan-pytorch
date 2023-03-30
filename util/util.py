"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
print('Loading util/util.py')
import re
import importlib
import torch
import os
import torchvision

def display_batch(epoch, data_i, trainer, writer, iter_counter):
    losses = trainer.get_latest_losses()
    for k,v in losses.items():
        writer.add_scalar(k,v.mean().item(), iter_counter.total_steps_so_far)
    writer.write_console(epoch, iter_counter.epoch_iter, iter_counter.time_per_iter)
    num_print = min(4, data_i['input'].size(0))
    infer_out, _ = trainer.pix2pix_model.forward(data_i, mode='inference')
    infer_out = (infer_out+1)/2
    infer_out = infer_out.clamp(0,1)
    torchvision.utils.save_image(infer_out[:num_print], 'output.png', nrow=1, normalize=True, range=(0,1))


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    net.cuda()

def load_network_path(net, save_path):
    weights = torch.load(save_path)
    new_dict = {}
    for k,v in weights.items():
        #if k.startswith("module.conv16") or k.startswith("module.conv17"):
        #    continue
        if k.startswith("module."):
            k=k.replace("module.","")
        new_dict[k] = v
    net.load_state_dict(new_dict, strict=False)
    #net.load_state_dict(new_dict)
    return net


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    print("==============load path: =================")
    print(save_path)
    new_dict = {}
    for k,v in weights.items():
        if k.startswith("module."):
            k=k.replace("module.","")
        new_dict[k] = v
    net.load_state_dict(new_dict, strict=False)
    return net
