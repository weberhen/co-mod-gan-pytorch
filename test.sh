python test.py \
	--mixing 0 \
	--batchSize 1 \
	--nThreads 1 \
	--name comod_places \
	--dataset_mode testimage \
	--image_dir /root/datasets_raid/zillow/panos \
	--mask_dir /root/datasets_raid/zillow/tripod_masks \
        --output_dir ./zillow \
	--load_size 128 \
	--crop_size 128 \
	--z_dim 512 \
	--model comod \
	--netG comodgan \
	--which_epoch epoch18_step490002 \
	--preprocess_mode resize \
	${EXTRA} \
