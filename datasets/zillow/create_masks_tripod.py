import numpy as np
import os
import cv2
from tqdm import tqdm
import multiprocessing

root_folder = '/root/datasets_raid/zillow/panos'

# list all files ending with .jpg
files = [f for f in os.listdir(root_folder) if f.endswith('.jpg')]

# create a folder for masks
mask_folder = '/root/datasets_raid/zillow/tripod_masks'

os.makedirs(mask_folder, exist_ok=True)

def tasklet(file):
# for file in tqdm(files):
    # read image
    img = cv2.imread(os.path.join(root_folder, file))
    # create mask
    # resize image to 128x128
    img = cv2.resize(img, (128, 128))
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[-30:, :] = 255
    # save mask
    cv2.imwrite(os.path.join(mask_folder, file), mask)
    a=1

# create a pool of as many processes as there are CPUs
pool = multiprocessing.Pool()
# run the tasklet on each file
pool.map(tasklet, files)
# close the pool
pool.close()