import os
import glob

folder_path = '/root/codes/co-mod-gan-pytorch'  # replace with the path to your folder

# Get a list of all files in the folder
file_paths = glob.glob(os.path.join(folder_path, '**/*'), recursive=True)

# ignore hidden files
file_paths = [f for f in file_paths if not f.startswith('.')]
# ignore png files
file_paths = [f for f in file_paths if not f.endswith('.png')]
# and pkl
file_paths = [f for f in file_paths if not f.endswith('.pkl')]
# and pth
file_paths = [f for f in file_paths if not f.endswith('.pth')]
# and pyc
file_paths = [f for f in file_paths if not f.endswith('.pyc')]
# and md
file_paths = [f for f in file_paths if not f.endswith('.md')]

# Loop through each file and count the number of lines
total_lines = 0
for file_path in file_paths:
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines
            print(f"{file_path}: {lines} lines")

print(f"Total lines: {total_lines}")
print(f"Total files: {len(file_paths)}")
