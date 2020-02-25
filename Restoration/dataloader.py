import numpy as np
from PIL import Image
import os as os
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms.functional as TF
import os
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import collections
from os.path import join as pjoin
from random import randint
import random


class dataloader(Dataset):

    def __init__(self, root_dir, transform):
        
        self.root_dir = os.path.expanduser(root_dir)
        path = pjoin(self.root_dir,'cityscapes/leftImg8bit/train') #change
        self.file_list = []
        for folders in os.listdir(path):
            for files in os.listdir(pjoin(path,folders)):
                self.file_list.append(pjoin(folders, files.rstrip()))
        self.transform = transform
        #self.angle_transform = transforms.ToTensor()
        
    def __getitem__(self, index):
        img =  Image.open(pjoin(self.root_dir,'cityscapes/leftImg8bit/train',self.file_list[index])).convert("RGB") #change
        timg =  Image.open(pjoin(self.root_dir,'tcityscapes/leftImg8bit/train',self.file_list[index])).convert("RGB") #change
        #print(pjoin(self.root_dir,'images',self.file_list[index]), pjoin(self.root_dir,'timages',self.file_list[index]) ,target)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(512, 512))
        img = TF.crop(img, i, j, h, w)
        timg = TF.crop(timg, i, j, h, w)
        ##
        img = self.transform(img)
        timg = self.transform(timg)
        
        return {'Img':img,'tImg': timg}

    def __len__(self):
        return len(self.file_list)

class vdataloader(Dataset):

    def __init__(self, root_dir, transform):
        
        self.root_dir = os.path.expanduser(root_dir)
        path = pjoin(self.root_dir,'cityscapes/leftImg8bit/val') #change
        self.file_list = []
        for folders in os.listdir(path):
            for files in os.listdir(pjoin(path,folders)):
                self.file_list.append(pjoin(folders, files.rstrip()))
        self.transform = transform
        #self.angle_transform = transforms.ToTensor()
        
    def __getitem__(self, index):

        img =  Image.open(pjoin(self.root_dir,'cityscapes/leftImg8bit/val',self.file_list[index])).convert("RGB") #change
        timg =  Image.open(pjoin(self.root_dir,'tcityscapes/leftImg8bit/val',self.file_list[index])).convert("RGB") #change
        
        img = img.resize((1024, 512), Image.ANTIALIAS)
        timg = timg.resize((1024, 512), Image.ANTIALIAS)
        #print(pjoin(self.root_dir,'images',self.file_list[index]), pjoin(self.root_dir,'timages',self.file_list[index]) ,target)
        # Random crop
        #i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(512, 1024))
        #img = TF.crop(img, i, j, h, w)
        #timg = TF.crop(timg, i, j, h, w)
        ##
        img = self.transform(img)
        timg = self.transform(timg)
        
        return {'Img':img,'tImg':timg, 'Path': self.file_list[index]}

    def __len__(self):
        return len(self.file_list)