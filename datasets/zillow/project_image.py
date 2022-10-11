import cv2
import numpy as np

img = '/root/codes/co-mod-gan-pytorch/zillow_debug/images/my_house.png'

img_zeros = np.zeros((256, 256, 3), dtype=np.uint8)

# resize img to 128x128
img = cv2.imread(img)
img = cv2.resize(img, (80, 80))

# paste img to img_zeros in the center
img_zeros[88:168, 88:168, :] = img

# save img_zeros
cv2.imwrite('/root/codes/co-mod-gan-pytorch/zillow_debug/images/my_house_256.png', img_zeros)

# make mask
mask = np.ones((256, 256, 3), dtype=np.uint8) * 255

# put 255 in the center of mask
mask[88:168, 88:168, :] = 0

# save mask
cv2.imwrite('/root/codes/co-mod-gan-pytorch/zillow_debug/masks/my_house_mask.png', mask)