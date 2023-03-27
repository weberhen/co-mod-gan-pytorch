import random 
import os

root_folder = '/root/datasets_ssd/zillow/panos'
output_folder = '/root/datasets_ssd/zillow/panos_split'

os.makedirs(output_folder, exist_ok=True)

# Get all the files
files = os.listdir(root_folder)
# append the full path
files = [os.path.join(root_folder, f) for f in files]
# remove ending .jpg
files = [f[:-4] for f in files]
# randomize the order
random.shuffle(files)

# Split into train and valid
perc_train = 0.8
train_files = files[:int(len(files)*perc_train)]
valid_files = files[int(len(files)*perc_train):]

# Write the files to disk
with open(os.path.join(output_folder, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_files))

with open(os.path.join(output_folder, 'valid.txt'), 'w') as f:
    f.write('\n'.join(valid_files))