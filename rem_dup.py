files = ["test.py", "data/base_dataset.py", "network/zillow_dataset.py", "util/util.py", "util/coco.py", "options/base_options.py", "options/train_options.py", "models/networks/base_network.py", "models/networks/loss.py", "models/networks/architecture.py", "models/networks/discriminator.py", "models/networks/op/fused_act.py", "models/networks/op/upfirdn2d.py", "models/networks/sync_batchnorm/batchnorm.py", "models/networks/sync_batchnorm/comm.py", "models/networks/sync_batchnorm/replicate.py", "models/networks/stylegan2.py", "models/networks/co_mod_gan.py", "models/networks/generator.py", "models/create_mask.py", "models/comod_model.py", "data/base_dataset.py", "data/testimage_dataset.py", "util/util.py", "util/coco.py", "data/base_dataset.py", "options/base_options.py", "options/train_options.py", "util/iter_counter.py", "network/zillow_dataset.py", "train.py", "models/networks/base_network.py", "models/networks/loss.py", "models/networks/architecture.py", "models/networks/discriminator.py", "models/networks/op/fused_act.py", "models/networks/op/upfirdn2d.py", "models/networks/sync_batchnorm/batchnorm.py", "models/networks/sync_batchnorm/comm.py", "models/networks/sync_batchnorm/replicate.py", "models/networks/stylegan2.py", "models/networks/co_mod_gan.py", "models/networks/generator.py", "models/create_mask.py", "models/comod_model.py", "data/base_dataset.py", "data/testimage_dataset.py", "trainers/stylegan2_trainer.py"]

unique_files = list(set(files))
# sort
unique_files.sort()
for file in unique_files:
    print(file)

