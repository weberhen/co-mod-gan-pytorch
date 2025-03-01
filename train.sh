export CXX="g++"
python train.py \
	--batchSize 8 \
	--nThreads 8 \
	--name comod_places \
	--train_image_dir /root/datasets_raid/zillow/panos \
	--train_image_list /root/datasets_raid/zillow/panos_split/train.txt \
	--train_image_postfix '.jpg' \
	--val_image_dir /root/datasets_raid/zillow/panos \
	--val_image_list /root/datasets_raid/zillow/panos_split/valid.txt \
	--val_mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 \
	--load_size 128 \
	--crop_size 128 \
	--z_dim 512 \
	--validation_freq 10000 \
	--niter 50 \
	--dataset_mode trainimage \
	--trainer stylegan2 \
	--dataset_mode_train trainimage \
	--dataset_mode_val valimage \
	--model comod \
	--netG comodgan \
	--netD comodgan \
	--no_l1_loss \
	--no_vgg_loss \
	--gpu_ids 0 \
	--preprocess_mode resize \
	--continue_train \
	--which_epoch epoch1_step280008 \
	$EXTRA
