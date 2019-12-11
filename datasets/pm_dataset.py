from PIL import Image
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import numpy as np
from medpy.io import load, save
from PIL import Image
import cv2
import glob
class traindataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.mode = mode
        self.train_data = []
        self.test_data  = []
        self.val_data   = []
        self.train_name = []
        self.train_root = list(np.genfromtxt(self.root + "full_train.txt", dtype='str'))
        self.train_label = np.genfromtxt(self.root + "full_label.txt", dtype=int)


        if self.mode =='train':
            # self.train_root = [self.train_root[idx] for idx in [0, -1]]
            # self.train_label = [self.train_label[idx] for idx in [0, -1]]
            # print(self.train_root)
            for item in self.train_root:
                img = Image.open(item)
                img = img.convert('RGB')
                self.train_data.append(img)
                self.train_name.append(item.split("/")[-1])
            assert len(self.train_label) == len(self.train_data)
            print ('=> Total Train: ', len(self.train_data) , " PM/NON-PM images ")

        elif self.mode == 'val':
            self.val_name = []
            self.test_root = list(np.genfromtxt(self.root + "test.txt", dtype='str'))
            self.val_label = list(np.genfromtxt(self.root + "test_label.txt", dtype=int))

            # self.test_root += list(np.genfromtxt(self.root+'train.txt', dtype='str'))
            # self.test_root = list(set(self.test_root)-set(self.train_root))
            # self.val_label = [1 if item.split("/")[-1][0] == "P" else 0 for item in self.test_root]


            for item in self.test_root:
                img = Image.open(item)
                img = img.convert('RGB')
                self.val_data.append(img)
                self.val_name.append(item.split("/"))

            assert len(self.val_data) == len(self.val_label)
            print ('=> Total Val: ', len(self.val_data) , " PM/NON-PM images ")


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.mode == 'train':
            img, label, name = self.train_data[index], self.train_label[index], self.train_name[index]
        elif self.mode == 'val':
            img, label, name = self.val_data[index], self.val_label[index], self.val_name[index]
        # elif self.mode =='test':
        #     img, label, name = self.test_data[index], self.test_label[index], self.test_name[index]

        img = self.transform(img)
        return img, label, name

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        # else:
        #     return len(self.test_data)