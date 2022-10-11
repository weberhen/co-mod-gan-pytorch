python test.py \
	--mixing 0 \
	--batchSize 1 \
	--nThreads 1 \
	--name comod-places-512 \
	--dataset_mode testimage \
	--image_dir /root/datasets_raid/zillow/panos \
	--mask_dir /root/datasets_raid/zillow/tripod_masks \
        --output_dir ./zillow \
	--load_size 512 \
	--crop_size 512 \
	--z_dim 512 \
	--model comod \
	--netG comodgan \
	--which_epoch co-mod-gan-places2-050000 \
	--preprocess_mode resize \
	${EXTRA} \
