import numpy as np
from PIL import Image
import torch

# what is a transform??
from torchvision import datasets, transforms
import os

# ??
from dataloaders import cifar_info

# creating a custom dataset by subclassing and overriding the Dataset class in torch.utils.data
# has funcns for length and get_item
class DummyDataset(torch.utils.data.Dataset):

    # x- image, y- label, trsf= transform, pretrsf= pre-transform, super_y= ??
    def __init__(self, x, y, trsf, pretrsf = None, imgnet_like = False, super_y = None):
        self.x, self.y = x, y
        self.super_y = super_y

        # transforms to be applied before and after conversion to imgarray
        self.trsf = trsf
        self.pretrsf = pretrsf

        # imgnet comes as array
        # if not from imgnet, needs to be converted to imgarray first
        self.imgnet_like = imgnet_like

    # since images are square, it only needs the width i guess
    def __len__(self):
        return self.x.shape[0]

    # return x[idx], y[idx], super_y[idx] after converting to array and applying transforms
    def __getitem__(self, idx):
        # does the only step thats required i guess
        x, y = self.x[idx], self.y[idx]
        # if super_y has something, return its idx-th element too
        if self.super_y is not None: super_y = self.super_y[idx]

        # apply a pre-transform if necessary
        if(self.pretrsf is not None): x = self.pretrsf(x)    
        
        # convert to array if necessary (i.e. if its not imgnet-like)
        if(not self.imgnet_like): x = Image.fromarray(x)
        
        # apply a post-transform i guess
        x = self.trsf(x)

        # return {x, y, super_y(if it exists)} 
        if self.super_y is not None: return x, y, super_y
        else: return x, y

# creating a custom dataset AGAIN, but this time the most simple
# has funcns for length and get_item - extremely simplistic and straightforward
class DummyArrayDataset(torch.utils.data.Dataset):

    # no transforms or supers or array-izers - just simple
    def __init__(self, x, y):
        self.x, self.y = x, y

    # usual stuff, getting the width
    def __len__(self):
        return self.x.shape[0]

    # simple return the idx-th item. [no transforms, no array-izing or supers]
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        return x, y

# return a collection of _get_datasets from the list
def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]

# the real deal of previous function
# returns dataset(object)s of {cifar10,cifar100,tinyimgnet}
def _get_dataset(dataset_name):
    # lower_case dataset_name and strip it of any spaces at the end or beginning
    dataset_name = dataset_name.lower().strip()

    # straightforward returning the appropriate dataset(object)
    if dataset_name == "cifar10": return iCIFAR10
    elif dataset_name == "cifar100": return iCIFAR100
    elif dataset_name == "tinyimagenet": return iImgnet
    elif dataset_name == "fruit": return iFruit
    else: raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

# a class created for nothing ??
# contains {base_dataset, train_transforms, common_transforms, class_order}
class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iImgnet(DataHandler):

    def open_and_convert_to_rgb(image_path):
        return Image.open(image_path[0]).convert('RGB')

    base_dataset = datasets.ImageFolder

    top_transforms = [
        open_and_convert_to_rgb,
    ]

    train_transforms = [
        transforms.RandomCrop(64, padding=4),           
        transforms.RandomHorizontalFlip() #,
        #transforms.ColorJitter(brightness=63 / 255)
    ]

    common_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
    ]

    class_order = [                                                                     
        i for i in range(200)
    ]


#Dataloader for fruit class
class iFruit(DataHandler):
    # No. of fruits
    n_fruits = 2
    img_size = 64

    # fruit dataset folder structure should be like:
    """
    - apple_unripe
        - images
            - image1.png
            - image2.png
            ....

    - apple_earlyripe
        - images
            - image1.png
            - image2.png
            ....

    .... 
    - apple_bad
        - images
            - image1.png
            - image2.png
            ....

    You can generalize this for all the fruits.
    """

    def open_and_convert_to_rgb(image_path):
        return Image.open(image_path[0]).convert('RGB')

    base_dataset = datasets.ImageFolder

    top_transforms = [
        open_and_convert_to_rgb,
    ]

    train_transforms = [
        transforms.RandomCrop(img_size, padding=4),           
        transforms.RandomHorizontalFlip() #,
        #transforms.ColorJitter(brightness=63 / 255)
    ]

    common_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
    ]

    class_order = [                                                  
        i for i in range(5*n_fruits) 
    ]

# CIFAR10 with only 10 classes
# subset of DataHandler
class iCIFAR10(DataHandler):
    # get an object
    base_dataset = datasets.cifar.CIFAR10
    # get the big damn class from 'cifar_info.py'
    base_dataset_hierarchy = cifar_info.CIFAR10

    # nothing
    top_transforms = [
    ]

    # crop, flip, colorjitting
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]

    # toTensor, normalize
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

# CIFAR100 with 100 classes
class iCIFAR100(iCIFAR10):
    # 
    base_dataset = datasets.cifar.CIFAR100
    base_dataset_hierarchy = cifar_info.CIFAR100

    # toTensor, normalize
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    # update: class order can now be chosen randomly since it just depends on seed
    class_order = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]   ## some random class order

    class_order_super = [4, 95, 55, 30, 72, 73, 1, 67, 32, 91, 62, 92, 70, 54, 82, 10, 61, 28, 9, 16, 53,
        83, 51, 0, 57, 87, 86, 40, 39, 22, 25, 5, 94, 84, 20, 18, 6, 7, 14, 24, 88, 97,
        3, 43, 42, 17, 37, 12, 68, 76, 71, 60, 33, 23, 49, 38, 21, 15, 31, 19, 75, 66, 34,
        63, 64, 45, 99, 26, 77, 79, 46, 98, 11, 2, 35, 93, 78, 44, 29, 27, 80, 65, 74, 50,
        36, 52, 96, 56, 47, 59, 90, 58, 48, 13, 8, 69, 81, 41, 89, 85
    ]   ## parent-wise split (increment 5 classes with all classes belong to one of the parent class)
    # for example starting 4, 95, 55, 30, 72 belong to super class of aquatic animals


"""
CIFAR 100 classes with order. compare the class_order with below

0: apple
1: aquarium_fish
2: baby
3: bear
4: beaver
5: bed
6: bee
7: beetle
8: bicycle
9: bottle
10: bowl
11: boy
12: bridge
13: bus
14: butterfly
15: camel
16: can
17: castle
18: caterpillar
19: cattle
20: chair
21: chimpanzee
22: clock
23: cloud
24: cockroach
25: couch
26: crab
27: crocodile
28: cup
29: dinosaur
30: dolphin
31: elephant
32: flatfish
33: forest
34: fox
35: girl
36: hamster
37: house
38: kangaroo
39: keyboard
40: lamp
41: lawn_mower
42: leopard
43: lion
44: lizard
45: lobster
46: man
47: maple_tree
48: motorcycle
49: mountain
50: mouse
51: mushroom
52: oak_tree
53: orange
54: orchid
55: otter
56: palm_tree
57: pear
58: pickup_truck
59: pine_tree
60: plain
61: plate
62: poppy
63: porcupine
64: possum
65: rabbit
66: raccoon
67: ray
68: road
69: rocket
70: rose
71: sea
72: seal
73: shark
74: shrew
75: skunk
76: skyscraper
77: snail
78: snake
79: spider
80: squirrel
81: streetcar
82: sunflower
83: sweet_pepper
84: table
85: tank
86: telephone
87: television
88: tiger
89: tractor
90: train
91: trout
92: tulip
93: turtle
94: wardrobe
95: whale
96: willow_tree
97: wolf
98: woman
99: worm
"""

