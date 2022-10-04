export CXX="g++"
python train.py \
	--batchSize 2 \
	--nThreads 2 \
	--name comod_places \
	--train_image_dir /root/datasets_raid/zillow/panos \
	--train_image_list /root/datasets_raid/zillow/panos_split/train.txt \
	--train_image_postfix '.jpg' \
	--val_image_dir /root/datasets_raid/zillow/panos \
	--val_image_list /root/datasets_raid/zillow/panos_split/valid.txt \
	--val_mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 \
	--load_size 512 \
	--crop_size 256 \
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
	--preprocess_mode scale_shortside_and_crop \
	$EXTRA
