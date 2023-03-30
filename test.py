"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
print('Loading test.py')
import cv2
from tqdm import tqdm
import numpy as np
import torch
from data.zillow_dataset import ZillowDataset
from options.train_options import TrainOptions
import models

opt = TrainOptions().parse()

dataloader = torch.utils.data.DataLoader(
    ZillowDataset(opt),
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=0)


model = models.create_model(opt)
model.eval()

for i, data_i in tqdm(enumerate(dataloader)):
    with torch.no_grad():
        generated,_ = model(data_i, mode='inference')
    generated = torch.clamp(generated, -1, 1)
    generated = (generated+1)/2*255
    generated = generated.cpu().numpy().astype(np.uint8)
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        pred_im = generated[b].transpose((1,2,0))
        print('process image... %s' % img_path[b])
        cv2.imwrite(img_path[b], pred_im[:,:,::-1])
