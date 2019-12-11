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
from PIL import Image

class traindataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.mode = mode
        self.train_data = []
        self.test_data  = []
        self.val_data   = []
        self.train_name = []

        if self.mode =='train':
            self.train_root = list(np.genfromtxt(self.root + "full_train.txt", dtype='str'))
            self.train_label = list(np.genfromtxt(self.root + "full_label.txt", dtype=int))

            # select_10data = list(range(0,20))[::2]
            # self.train_root = [self.train_root[idx] for idx in select_10data]
            # self.train_label = [self.train_label[idx] for idx in select_10data]

            for item in self.train_root:
                img = Image.open(self.root + item)
                img = img.convert('RGB')
                self.train_data.append(img)
                self.train_name.append(item)
            assert len(self.train_label) == len(self.train_data)
            print ('=> Total Train: ', len(self.train_data) , " AMD/NON-AMD images ")

        elif self.mode == 'val':
            self.val_name = []
            test_root = list(np.genfromtxt(self.root + "test.txt", dtype='str'))
            self.val_label = list(np.genfromtxt(self.root + "testlabel.txt", dtype=int))

            # test_root2 = list(np.genfromtxt(self.root + "full_train.txt", dtype='str'))
            # val_label2 = list(np.genfromtxt(self.root + "full_label.txt", dtype=int))
            # test_root = test_root + test_root2
            # self.val_label = self.val_label + val_label2

            for item in test_root:
                img = Image.open(self.root + item)
                img = img.convert('RGB')
                self.val_data.append(img)
                self.val_name.append(item)

            assert len(self.val_data) == len(self.val_label)
            print ('=> Total Val: ', len(self.val_data) , " AMD/NON-AMD images ")
        #
        # elif self.mode == 'test':
        #     self.test_name = []
        #     test_root = np.genfromtxt(self.root + "test.txt", dtype='str')
        #     self.test_label = np.genfromtxt(self.root + "testlabel.txt", dtype=int)
        #
        #     for item in test_root:
        #         img = Image.open(self.root + item)
        #         img = img.convert('RGB')
        #         self.test_data.append(img)
        #         self.test_name.append(item)
        #
        #     assert len(self.test_data) == len(self.test_label)
        #     print ('=> Total Test: ', len(self.test_data) , " AMD/NON-AMD images ")

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