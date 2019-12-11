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
import glob
import csv

class traindataset(data.Dataset):
    def __init__(self, root, mode, transform=None, num_class=5, multitask=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.mode = mode
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.name =[]
        self.num_class = num_class
        self.multitask = multitask


        files = glob.glob(self.root+'/images/trainset/*.jpg')
        # val_idx = np.array([219,  32, 278, 142, 211,  55, 409, 253, 401, 116, 408, 133, 347,
        #                    128, 218, 305, 265, 368, 128, 146, 302, 108, 332, 173,  34, 253,
        #                    375, 162, 280, 199,  89, 381, 150,  95, 354, 163, 273, 327, 125,
        #                    283, 265, 260, 163, 384,  75, 291, 257, 400, 410, 314, 250, 122,
        #                     68, 300, 305,  40, 276, 253, 226, 385,  47, 242, 205, 387, 187,
        #                    372,   6, 222,  41,  37, 234, 286, 171, 167, 159,  36, 355, 157,
        #                    396, 135])
        # self.val_root = [files[idx] for idx in val_idx]
        self.train_root  = files

        if self.mode =='train':
            dictLabels_DR, dictLabels_DME = self.load_csv(self.root + '/labels/trainset.csv')
            for each_one in self.train_root:
                img = Image.open(each_one)
                img = img.convert('RGB')
                if self.multitask:
                    label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1][:-4] in v]
                    label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1][:-4] in v]
                    self.train_label.append([int(label_DR[0]), int(label_DME[0])])
                else:
                    if self.num_class == 5:
                        label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1][:-4] in v]
                    else:
                        label_DR = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1][:-4] in v]
                    self.train_label.append(int(label_DR[0]))

                assert len(label_DR) == 1
                # label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1][:-4] in v]
                self.train_data.append(img)
                self.name.append(each_one.split("/")[-1][:-4])
            assert len(self.train_label) == len(self.train_root)

            if self.multitask:
                print('=> Total Train: ', len(self.train_root), " Multi-Task images ")
            else:
                if self.num_class == 5:
                    print ('=> Total Train: ', len(self.train_root) , " DR images ")
                else:
                    print('=> Total Train: ', len(self.train_root), " DME images ")

        elif self.mode == 'val':
            self.test_root = glob.glob(self.root + '/images/testset/*.jpg')
            dictLabels_DR, dictLabels_DME = self.load_csv(self.root + '/labels/testset.csv')
            for item in self.test_root:
                img = Image.open(item)
                img = img.convert('RGB')
                if self.multitask:
                    label_DR = [k for k, v in dictLabels_DR.items() if item.split("/")[-1][:-4] in v]
                    label_DME = [k for k, v in dictLabels_DME.items() if item.split("/")[-1][:-4] in v]
                    self.test_label.append([int(label_DR[0]), int(label_DME[0])])
                else:
                    if self.num_class == 5:
                        label_DR = [k for k, v in dictLabels_DR.items() if item.split("/")[-1][:-4] in v]
                    else:
                        label_DR = [k for k, v in dictLabels_DME.items() if item.split("/")[-1][:-4] in v]
                    self.test_label.append(int(label_DR[0]))

                assert len(label_DR) == 1
                self.test_data.append(img)
                self.name.append(item.split("/")[-1][:-4])
            assert len(self.test_root) == len(self.test_label)

            if self.multitask:
                print('=> Total Test: ', len(self.test_root), " Multi-Task images ")
            else:
                if self.num_class == 5:
                    print ('=> Total Test: ', len(self.test_root) , " DR images ")
                else:
                    print('=> Total Test: ', len(self.test_root), " DME images ")

    def load_csv(self, path):

        dictLabels_DR = {}
        dictLabels_DME = {}
        with open(path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label1 = row[1]
                label2 = row[2]

                if label1 in dictLabels_DR.keys():
                    dictLabels_DR[label1].append(filename)
                else:
                    dictLabels_DR[label1] = [filename]

                if label2 in dictLabels_DME.keys():
                    dictLabels_DME[label2].append(filename)
                else:
                    dictLabels_DME[label2] = [filename]

            return dictLabels_DR, dictLabels_DME


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """


        if self.mode == 'train':
            # img = Image.open(self.train_root[index])
            # img = img.convert('RGB')
            img = self.train_data[index]
            label , name  = self.train_label[index], self.name[index]
        elif self.mode == 'val':
            # img = Image.open(self.test_root[index])
            # img = img.convert('RGB')
            img = self.test_data[index]
            label , name = self.test_label[index], self.name[index]

        img = self.transform(img)


        return img, label, name

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_root)
        elif self.mode == 'val':
            return len(self.test_root)


if __name__ == '__main__':
    import cv2
    root = "/home/xmli/datasets/IDRiD/Segmentation/"
    import os.path
    for i in range(1,55):
        idx = ("{:02d}".format(i))
        img_root = root + "OriginalImages/Trainset/IDRiD_"+str(idx)+".jpg"
        label_he_root = root + "GroundTruths/Trainset/Haemorrhages/IDRiD_"+str(idx)+"_HE.tif"
        label_ex_root = root + "GroundTruths/Trainset/HardExudates/IDRiD_"+str(idx)+"_EX.tif"
        label_ma_root = root + "GroundTruths/Trainset/Microaneurysms/IDRiD_"+str(idx)+"_MA.tif"
        label_od_root = root + "GroundTruths/Trainset/OpticDisc/IDRiD_"+str(idx)+"_OD.tif"
        label_se_root = root + "GroundTruths/Trainset/SoftExudates/IDRiD_"+str(idx)+"_SE.tif"

        img = cv2.imread(img_root)

        if os.path.isfile(label_he_root):
            label_he = cv2.imread(label_he_root, cv2.IMREAD_GRAYSCALE)
            contours_label = cv2.findContours(label_he, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_label[1], -1, (0, 255, 255), 3)
        if os.path.isfile(label_ex_root):
            label_ex = cv2.imread(label_ex_root, cv2.IMREAD_GRAYSCALE)
            contours_label = cv2.findContours(label_ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_label[1], -1, (255, 0, 0), 3)
        if os.path.isfile(label_ma_root):
            label_ma = cv2.imread(label_ma_root, cv2.IMREAD_GRAYSCALE)
            contours_label = cv2.findContours(label_ma, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_label[1], -1, (0, 255, 0), 3)
        if os.path.isfile(label_od_root):
            label_od = cv2.imread(label_od_root, cv2.IMREAD_GRAYSCALE)
            contours_label = cv2.findContours(label_od, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_label[1], -1, (255, 255, 0), 3)
        if os.path.isfile(label_se_root):
            label_se = cv2.imread(label_se_root, cv2.IMREAD_GRAYSCALE)
            contours_label = cv2.findContours(label_se, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_label[1], -1, (0, 0, 255), 3)
        cv2.imwrite(root + 'visual_trainset/'+ img_root.split("/")[-1], img)

