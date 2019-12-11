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
    root = "/home/xmli/datasets/IDRiD/Localization/"
    import os.path
    import csv
    import cv2
    from PIL import Image, ImageDraw

    length = 50
    # load label
    Fovea_label = root + "Groundtruths/FoveaCenterLocation/IDRiD_Fovea_Center_TrainingSetMarkups.csv"
    OD_label = root + "Groundtruths/OpticDiscCenterLocation/IDRiD_OD_Center_TrainingSet_Markups.csv"
    filename_list = []
    axis_x_list = []
    axis_y_list = []
    with open(Fovea_label) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename_list.append(row[0])
            axis_x_list.append(row[1])
            axis_y_list.append(row[2])

    OD_axis_x_list = []
    OD_axis_y_list = []
    with open(OD_label) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename_list.append(row[0])
            OD_axis_x_list.append(row[1])
            OD_axis_y_list.append(row[2])

    for i in range(0,413):
        idx = ("{:03d}".format(i+1))
        img_root = root + "OriginalImages/TrainingSet/IDRiD_"+str(idx)+".jpg"
        img = cv2.imread(img_root)
        cv2.line(img, (int(axis_x_list[i])-length, int(axis_y_list[i])),
                 (int(axis_x_list[i])+length, int(axis_y_list[i])), (0, 255, 0), 7)
        cv2.line(img, (int(axis_x_list[i]), int(axis_y_list[i])-length),
                 (int(axis_x_list[i]), int(axis_y_list[i])+length), (0, 255, 0), 7)

        cv2.line(img, (int(OD_axis_x_list[i]) - length, int(OD_axis_y_list[i])),
                 (int(OD_axis_x_list[i]) + length, int(OD_axis_y_list[i])), (255, 0, 0), 7)
        cv2.line(img, (int(OD_axis_x_list[i]), int(OD_axis_y_list[i])-length),
                 (int(OD_axis_x_list[i]), int(OD_axis_y_list[i])+length), (255, 0, 0), 7)
        cv2.imwrite(root + 'visual_trainset/' + img_root.split("/")[-1], img)
