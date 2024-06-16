import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

target_folder = './tiny-imagenet-200/val/'

val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

print(val_dict)
paths = glob.glob('./tiny-imagenet-200/val/images/*')
print(paths)
temp = []
for path in paths:
    k = path.split('\\')
    temp.append(k[0]+"/"+k[1])

paths=temp
print(paths)
print("------------------------------------------------------------------------------------------------------------------------------")
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
print("------------------------------------------------------------------------------------------------------------------------------")
        
        
print("------------------------------------------------------------------------------------------------------------------------------")
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)
print("------------------------------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------------------------")
rmdir('./tiny-imagenet-200/val/images')
print("------------------------------------------------------------------------------------------------------------------------------")
