print('Loading network/zillow_dataset.py')
from PIL import Image
from torchvision import transforms
import torch.utils.data
import h5py
import os

class ZillowDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.dataset_dir = args.train_image_dir
        self.in_size = args.load_size
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.load_entire_dataset_into_memory()

    def __len__(self):
        return len(self.gt_images) 
    
    def load_entire_dataset_into_memory(self):
        print('Loading entire dataset into memory')
        # # save self.gt_images as h5 file
        # with h5py.File('gt_images.h5', 'w') as hf:
        #     hf.create_dataset("gt_images",  data=self.gt_images)
        # load self.gt_images from h5 file
        with h5py.File(os.path.join(self.dataset_dir, 'gt_images.h5'), 'r') as hf:
            self.gt_images = hf['gt_images'][:]
        print('Loaded entire dataset into memory')

    def __getitem__(self, index):
        gt_image = self.gt_images[index]
        input_image = gt_image
        input_image = Image.fromarray(input_image)
        gt_image = Image.fromarray(gt_image)
        # make mask to be the same shape as input_image, zeros everywhere except for the center
        mask = torch.zeros_like(self.transform(input_image))[0,].unsqueeze(0)
        mask[:, self.in_size//4:self.in_size//4*3, self.in_size//4:self.in_size//4*3] = 1
        return {'input': self.transform(input_image).cuda(), 'gt': self.transform(gt_image).cuda(), 'mask': mask.cuda()}