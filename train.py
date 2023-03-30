"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from util.iter_counter import IterationCounter
from my_logger import Logger
from util.util import display_batch
from trainers import stylegan2_trainer
from data.zillow_dataset import ZillowDataset
import omegaconf

print('Loading train.py')

def training_loop():
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader_train, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            # train discriminator
            if not opt.freeze_D:
                trainer.run_discriminator_one_step(data_i, i)

            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i, i)

            if iter_counter.needs_displaying():
                display_batch(epoch, data_i, trainer, writer, iter_counter)
            if iter_counter.needs_validation():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('epoch%d_step%d'%
                        (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

if __name__ == '__main__':
    # parse options with omegaconf
    opt = omegaconf.OmegaConf.load('conf/default.yaml')
    
    # dataloader_train = data.create_dataloader(opt)
    dataloader_train = torch.utils.data.DataLoader(ZillowDataset(opt), batch_size=opt.batchSize, shuffle=True, num_workers=0)

    # create trainer for our model
    trainer = stylegan2_trainer.StyleGAN2Trainer(opt)
    model = trainer.pix2pix_model

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader_train))

    # create tool for visualization
    writer = Logger(f"output/{opt.name}")
    with open(f"output/{opt.name}/savemodel", "w") as f:
        f.writelines("n")

    trainer.save('latest')

    training_loop()