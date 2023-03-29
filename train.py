"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from my_logger import Logger
import torchvision
from trainers import create_trainer
from data.zillow_dataset import ZillowDataset

print('Loading train.py')

def display_batch(epoch, data_i):
    losses = trainer.get_latest_losses()
    for k,v in losses.items():
        writer.add_scalar(k,v.mean().item(), iter_counter.total_steps_so_far)
    writer.write_console(epoch, iter_counter.epoch_iter, iter_counter.time_per_iter)
    num_print = min(4, data_i['input'].size(0))
    infer_out, _ = trainer.pix2pix_model.forward(data_i, mode='inference')
    infer_out = (infer_out+1)/2
    infer_out = infer_out.clamp(0,1)
    torchvision.utils.save_image(infer_out[:num_print], 'output.png', nrow=1, normalize=True, range=(0,1))
    

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
                display_batch(epoch, data_i)
            if iter_counter.needs_validation():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('epoch%d_step%d'%
                        (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
                if dataloader_val is not None:
                    print("doing validation")
                    model.eval()
                    num = 0
                    
                    for ii, data_ii in enumerate(dataloader_val):
                        with torch.no_grad():
                            generated,_ = model(data_ii, mode='inference')
                            generated = generated.cpu()
                        gt = data_ii['input']
                        bsize = gt.size(0)
                        num += bsize
                    model.train()
        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

if __name__ == '__main__':

    # parse options
    opt = TrainOptions().parse()

    opt.dataset_mode_val = None
    # load the dataset
    if opt.dataset_mode_val is not None:
        dataloader_train, dataloader_val = data.create_dataloader_trainval(opt)
    else:
        # dataloader_train = data.create_dataloader(opt)
        dataloader_train = torch.utils.data.DataLoader(ZillowDataset(opt), batch_size=opt.batchSize, shuffle=True, num_workers=0)
        dataloader_val = None

    # create trainer for our model
    trainer = create_trainer(opt)
    model = trainer.pix2pix_model

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader_train))

    # create tool for visualization
    writer = Logger(f"output/{opt.name}")
    with open(f"output/{opt.name}/savemodel", "w") as f:
        f.writelines("n")

    trainer.save('latest')

    training_loop()