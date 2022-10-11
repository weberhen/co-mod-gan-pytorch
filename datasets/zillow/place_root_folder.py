import sys
import os

file = sys.argv[1]
string_to_replace = sys.argv[2]
replacement_string = sys.argv[3]
# file = '/root/datasets_raid/zillow/panos_split/train_test.txt'
# string_to_replace = '/root/datasets_raid/zillow/panos'
# replacement_string = '/test'

with open(file, "rt") as fin:
    with open(file[:-4]+'_out.txt', "wt") as fout:
        for line in fin:
            fout.write(line.replace(string_to_replace, replacement_string))

# overwrite the original file with the new one
os.rename(file[:-4]+'_out.txt', file)