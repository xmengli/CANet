import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from PIL import Image
import xlrd
import glob
import csv


class traindataset(data.Dataset):
    def __init__(self, root, mode, transform=None, num_class=5, multitask=False, args=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.mode = mode
        self.train_label = []
        self.test_label = []
        self.name =[]
        self.num_class = num_class
        self.multitask = multitask

        if self.mode == "train":
            self.train_root = glob.glob(self.root + "/train/train/*.jpeg")
            dictLabels_DR = self.load_csv(self.root + "/trainLabels.csv")
            for each_one in self.train_root:
                label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1][:-5] in v]
                self.train_label.append(int(label_DR[0]))
                assert len(label_DR) == 1
                self.name.append(each_one.split("/")[-1])
            assert len(self.train_label) == len(self.train_root)
            print('=> Total Train: ', len(self.train_root), " DR images ")
        elif self.mode == "val":
            self.test_root = glob.glob(self.root + "/train/train/*.jpeg")
            self.test_root = self.test_root[:30]
            dictLabels_DR = self.load_csv(self.root + "/trainLabels.csv")
            for each_one in self.test_root:
                label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1][:-5] in v]
                self.test_label.append(int(label_DR[0]))
                assert len(label_DR) == 1
                self.name.append(each_one.split("/")[-1])
            assert len(self.test_label) == len(self.test_root)
            print('=> Total Test: ', len(self.test_root), " DR images ")


    def load_csv(self, path):
        dictLabels_DR = {}
        with open(path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label1 = row[1]
                if label1 in dictLabels_DR.keys():
                    dictLabels_DR[label1].append(filename)
                else:
                    dictLabels_DR[label1] = [filename]
        return dictLabels_DR

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """


        if self.mode == 'train':
            img = Image.open(self.train_root[index])
            img = img.convert('RGB')
            label , name  = self.train_label[index], self.name[index]
        elif self.mode == 'val':
            img = Image.open(self.test_root[index])
            img = img.convert('RGB')
            label , name = self.test_label[index], self.name[index]

        img = self.transform(img)
        return img, label, name

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_root)
        elif self.mode == 'val':
            return len(self.test_root)