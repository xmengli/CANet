import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from PIL import Image
import glob
import xlrd
import numpy as np
import scipy.misc



class traindataset(data.Dataset):
    def __init__(self, root, mode, transform=None, num_class=5, multitask=False, args=None):
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


        # get file path
        # files = glob.glob(self.root+'/*/*.tif')
        # self.train_root  = files
        # get file name and label
        xls_files = glob.glob(self.root + '/*/*.xls')
        dictLabels_DR, dictLabels_DME = self.load_csv(xls_files)


        files = np.loadtxt(self.root + "file_list.txt", dtype=str)
        idx = np.loadtxt(self.root + "/10fold/"+str(args.fold_name)+".txt", dtype=int)
        print ("foldname", args.fold_name)
        # idx = [428, 42, 336, 1068, 246, 790, 157, 987, 96, 1176, 1084, 99, 718, 1143,
        #        1139, 175, 417, 1112, 736, 474, 643, 106, 521, 1142, 668, 971, 377, 580,
        #        383, 688, 431, 49, 346, 1198, 773, 62, 1183, 60, 644, 191, 945, 939, 48,
        #        466, 137, 424, 506, 1094, 901, 274, 416, 783, 763, 6, 974, 1072, 421, 583,
        #        854, 299, 17, 902, 12, 329, 85, 623, 292, 997, 287, 629, 537, 640, 51, 1024,
        #        684, 978, 935, 360, 924, 1189, 225, 731, 84, 942, 334, 885, 921, 890, 861, 837,
        #        401, 665, 349, 63, 1003, 1000, 500, 794, 1073, 1075, 345, 599, 973, 806, 272, 553,
        #        801, 479, 680, 770, 409, 1175, 179, 1144, 1145, 738, 308, 271, 1078, 778, 827, 395,
        #        11, 113, 224, 716, 1006, 30, 244, 4, 953, 605, 911, 423, 88, 875, 1088, 238, 250, 947,
        #        582, 857, 673, 1029, 1108, 655, 344, 436, 1030, 725, 887, 337, 705, 658, 621, 389,
        #        1041, 355, 1194, 1147, 757, 1096, 642, 280, 15, 788, 1177, 497, 103, 444, 29, 602,
        #        859, 362, 823, 155, 802, 110, 1101, 1129, 549, 307, 273, 1131, 775, 256, 231, 123,
        #        1126, 910, 126, 561, 43, 918, 767, 661, 1170, 130, 1125, 28, 405, 663, 57, 972, 650,
        #        257, 268, 1117, 737, 632, 706, 198, 111, 193, 715, 879, 1186, 140, 922, 114, 708,
        #        641, 89, 164, 27, 441, 829, 1052, 1076, 235, 888, 721, 366, 707, 1070, 624, 1079,
        #        464, 32, 1081]
        # idx = idx[:120]
        self.test_root = [files[idx_item] for idx_item in idx]
        self.train_root = list(set(files) - set(self.test_root))
        self.train_root = [self.root + item for item in self.train_root]
        self.test_root  = [self.root + item for item in self.test_root]


        # self.train_root = np.genfromtxt("train_root.txt", dtype=str)
        # self.test_root = np.genfromtxt("test_root.txt", dtype=str)
        # np.savetxt("train_root.txt", self.train_root, fmt="%s")
        # np.savetxt("test_root.txt", self.test_root, fmt="%s")
        # print ("test sample", self.test_root[0])

        if self.mode =='train':
            # self.train_root = [item.replace("/medical/", "/Documents/datasets/medical/") for item in self.train_root]
            for each_one in self.train_root:
                img = Image.open(each_one)
                img = img.convert('RGB')
                if self.multitask:
                    label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1] in v]
                    label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1] in v]
                    self.train_label.append([int(label_DR[0]), int(label_DME[0])])
                else:
                    if self.num_class == 2:
                        label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1] in v]
                    else:
                        label_DR = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1] in v]
                    self.train_label.append(int(label_DR[0]))

                assert len(label_DR) == 1
                # label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1][:-4] in v]
                self.train_data.append(img)
                self.name.append(each_one.split("/")[-1])
            assert len(self.train_label) == len(self.train_data)

            if self.multitask:
                print('=> Total Train: ', len(self.train_data), " Multi-Task images ")
            else:
                if self.num_class == 2:
                    print ('=> Total Train: ', len(self.train_data) , " DR images ")
                else:
                    print('=> Total Train: ', len(self.train_data), " DME images ")

        elif self.mode == 'val':
            # print(self.test_root)
            # self.test_root = [item.replace("/medical/", "/Documents/datasets/medical/") for item in self.test_root]

            for item in self.test_root:
                img = Image.open(item)
                img = img.convert('RGB')
                if self.multitask:
                    label_DR = [k for k, v in dictLabels_DR.items() if item.split("/")[-1] in v]
                    label_DME = [k for k, v in dictLabels_DME.items() if item.split("/")[-1] in v]
                    self.test_label.append([int(label_DR[0]), int(label_DME[0])])
                else:
                    if self.num_class == 2:
                        label_DR = [k for k, v in dictLabels_DR.items() if item.split("/")[-1] in v]
                    else:
                        label_DR = [k for k, v in dictLabels_DME.items() if item.split("/")[-1] in v]
                    self.test_label.append(int(label_DR[0]))
                assert len(label_DR) == 1
                self.test_data.append(img)
                self.name.append(item.split("/")[-1])
            assert len(self.test_data) == len(self.test_label)
            if self.multitask:
                print('=> Total Test: ', len(self.test_data), " Multi-Task images ")
            else:
                if self.num_class == 2:
                    print ('=> Total Test: ', len(self.test_data) , " DR images ")
                else:
                    print('=> Total Test: ', len(self.test_data), " DME images ")

    def load_csv(self, path):

        dictLabels_DR = {}
        dictLabels_DME = {}
        for per_path in path:
            # open xlsx
            xl_workbook = xlrd.open_workbook(per_path)
            xl_sheet = xl_workbook.sheet_by_index(0)
            for rowx in range(1, xl_sheet.nrows):
                cols = xl_sheet.row_values(rowx)
                filename = cols[0]
                label1 = int(cols[2])
                label2 = int(cols[3])

                if label1 < 2:
                    label1 = 0
                else:
                    label1 = 1

                if label1 in dictLabels_DR.keys():
                    dictLabels_DR[label1].append(filename)
                else:
                    dictLabels_DR[label1] = [filename]

                if label2 in dictLabels_DME.keys():
                    dictLabels_DME[label2].append(filename)
                else:
                    dictLabels_DME[label2] = [filename]

        # print (len(dictLabels_DR[0])) 546
        # print (len(dictLabels_DR[1])) 153
        # print (len(dictLabels_DR[2])) 247
        # print (len(dictLabels_DR[3])) 254
        #
        # print (len(dictLabels_DME[0])) 974
        # print (len(dictLabels_DME[1])) 75
        # print (len(dictLabels_DME[2])) 151
        return dictLabels_DR, dictLabels_DME

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """


        if self.mode == 'train':
            img, label , name  = self.train_data[index], self.train_label[index], self.name[index]
        elif self.mode == 'val':
            img, label , name = self.test_data[index], self.test_label[index], self.name[index]

        img = self.transform(img)


        return img, label, name

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.test_data)
