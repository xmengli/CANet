import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.cbam import *
import scipy.misc
from skimage.transform import resize

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

import torch
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None,
                 multitask=False, liu=False, chen=False, CAN_TS=False, crossCBAM=False,
                 crosspatialCBAM= False,  choice=""):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.multitask = multitask
        self.liu = liu
        self.chen = chen
        self.num_classes = num_classes
        self.CAN_TS = CAN_TS
        self.crossCBAM = crossCBAM
        self.crosspatialCBAM = crosspatialCBAM
        self.choice = choice

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        if not self.crossCBAM:
            self.layer5 = nn.Conv2d(2048, 300, kernel_size=3, padding=1, bias=False)
            self.bn5 = norm_layer(300)
            self.relu5= nn.ReLU(inplace=True)


        if self.liu:
            self.dropout = nn.Dropout(0.5)
            self.branch2  = nn.Linear(512 * block.expansion, 256)
            self.classifier1 = nn.Linear(512 * block.expansion+256, self.num_classes)
            self.classifier2 = nn.Linear(256, 3)
        elif self.chen:
            self.dropout = nn.Dropout(0.5)
            self.fc_out = nn.Linear(512 * block.expansion, 1024)
            self.classifier1_1 = nn.Linear(1024, 256)
            self.classifier1_2 = nn.Linear(256, 128)
            self.classifier1_3 = nn.Linear(128, self.num_classes)
            self.classifier2_1 = nn.Linear(1024, 256)
            self.classifier2_2 = nn.Linear(256, 128)
            self.classifier2_3 = nn.Linear(128, 3)
        elif self.crossCBAM:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam1 = CBAM(512 * block.expansion)
            self.branch_bam2 = CBAM(512 * block.expansion)
            self.classifier_dep1 = nn.Linear(512 * block.expansion, 1024)
            self.classifier_dep2 = nn.Linear(512 * block.expansion, 1024)
            self.branch_bam3 = CBAM(1024, no_spatial=True)
            self.branch_bam4 = CBAM(1024, no_spatial=True)
            self.classifier1 = nn.Linear(1024, self.num_classes)
            self.classifier2 = nn.Linear(1024, 3)
            self.classifier_specific_1 = nn.Linear(1024, self.num_classes)
            self.classifier_specific_2 = nn.Linear(1024, 3)

        elif self.crosspatialCBAM:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam1 = CBAM(512 * block.expansion)
            self.branch_bam2 = CBAM(512 * block.expansion)
            self.classifier_specific_1 = nn.Linear(512 * block.expansion, self.num_classes)
            self.classifier_specific_2 = nn.Linear(512 * block.expansion, 3)
            # self.conv_final1 = nn.Linear(512 * block.expansion, 1024)
            # self.conv_final2 = nn.Linear(512 * block.expansion, 1024)
            # self.bn_final1 = norm_layer(1024)
            # self.bn_final2 = norm_layer(1024)
            # self.relu_final = nn.ReLU(inplace=True)
            self.branch_bam3 = CBAM(2048)
            self.branch_bam4 = CBAM(2048)
            self.classifier1 = nn.Linear(2048, self.num_classes)
            self.classifier2 = nn.Linear(2048, 3)
        elif self.CAN_TS:
            self.dropout = nn.Dropout(0.3)
            self.branch_bam1 = CBAM(512 * block.expansion)
            self.branch_bam2 = CBAM(512 * block.expansion)
            self.classifier1 = nn.Linear(512 * block.expansion, self.num_classes)
            self.classifier2 = nn.Linear(512 * block.expansion, 3)
        elif self.multitask:
            # self.classifier1 = nn.Linear(512 * block.expansion, self.num_classes)
            # self.classifier2 = nn.Linear(512 * block.expansion, 3)
            self.classifier1 = nn.Linear(300, self.num_classes)
            self.classifier2 = nn.Linear(300, 3)
        elif self.num_classes == 2:
            self.classifier1 = nn.Linear(512 * block.expansion, self.num_classes)
        elif self.num_classes == 3:
            self.classifier2 = nn.Linear(512 * block.expansion, self.num_classes)
        elif self.num_classes == 5:
            self.classifier3 = nn.Linear(512 * block.expansion, 5)

        # self.classifier_specific_1 = nn.Linear(1024, self.num_classes)
        # self.classifier_specific_2 = nn.Linear(1024, 3)

        # self.branch_bam_dr1  = CBAM(256)
        # self.branch_bam_dme1 = CBAM(256)
        # self.branch_bam_dr2  = CBAM(512)
        # self.branch_bam_dme2 = CBAM(512)
        # self.branch_bam_dr3  = CBAM(1024)
        # self.branch_bam_dme3 = CBAM(1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # import cv2
        # arr = x[4]
        # print (arr.shape)
        # exit(0)
        # arr = arr - [-0.485/0.229, -0.456/0.224, -0.406/0.255]
        # arr = arr / [1/0.229, 1/0.224, 1/0.255]
        #
        # arr = arr[4, :, :, :].cpu().data.numpy()
        # arr = arr.permute(1,2,0)
        # print (arr.shape)
        # exit(0)
        # color_img = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        # scipy.misc.toimage(color_img, cmin=0.0, cmax=1.0).save(
        #     'visual_result/04_1,2/raw_img.jpg')

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x1_dr = self.avgpool(self.branch_bam_dr1(x))
        # x1_dme = self.avgpool(self.branch_bam_dme1(x))

        x = self.layer2(x)
        # x2_dr = self.avgpool(self.branch_bam_dr2(x))
        # x2_dme = self.avgpool(self.branch_bam_dme2(x))

        x = self.layer3(x)
        # x3_dr = self.avgpool(self.branch_bam_dr3(x))
        # x3_dme = self.avgpool(self.branch_bam_dme3(x))

        x = self.layer4(x)

        if not self.crossCBAM:
            x = self.relu5(self.bn5(self.layer5(x)))


        if self.crossCBAM:
            x = self.dropout(x)

            # x_dr, scale = self.branch_bam1(x)
            # for i in range(0, 40):
            #     img = resize(scale[i, 0, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #                  cval=0, clip=True, preserve_range=True)
            #     scipy.misc.imsave('visual_result/DR_attention_map_' + str(i) + '.jpg', img)
            # x_dme, scale2 = self.branch_bam2(x)
            # for i in range(0, 40):
            #     img = resize(scale2[i, 0, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #                  cval=0, clip=True, preserve_range=True)
            #     scipy.misc.imsave('visual_result/DME_attention_map_' + str(i) + '.jpg', img)
            #
            # for i in range(0, 40):
            #     img = resize(x_dr[i, 0, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #                  cval=0, clip=True, preserve_range=True)
            #     scipy.misc.imsave('visual_result/DR_features0_' + str(i) + '.jpg', img)
            #
            # for i in range(0, 40):
            #     img = resize(x_dr[i, 2, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #                  cval=0, clip=True, preserve_range=True)
            #     scipy.misc.imsave('visual_result/DR_features2_' + str(i) + '.jpg', img)
            #
            #
            # for i in range(0, 40):
            #     img = resize(x_dme[i, 0, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #                  cval=0, clip=True, preserve_range=True)
            #     scipy.misc.imsave('visual_result/DME_features_' + str(i) + '.jpg', img)
            #
            #
            # for i in range(0, 40):
            #     img = resize(x_dme[i, 2, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #                  cval=0, clip=True, preserve_range=True)
            #     scipy.misc.imsave('visual_result/DR_features2_' + str(i) + '.jpg', img)
            #
            # exit(0)
            # # #
            # for i in range(0,40):
            #     img = resize(self.branch_bam1(x)[5, i, :, :].cpu().data.numpy(), (224, 224), order=3,
            #                  mode='constant', cval=0, clip=True, preserve_range=True)
            #     ori_img = np.zeros((350, 350), dtype='float32')
            #     ori_img[63:287, 63:287] = img
            #     scipy.misc.imsave('visual_result/messidor_05/DR_spatial_attention' + str(i) + '.jpg', ori_img)
            # for i in range(0, 40):
            #     img = resize(self.branch_bam2(x)[5, i, :, :].cpu().data.numpy(), (224, 224), order=3,
            #                  mode='constant', cval=0, clip=True, preserve_range=True)
            #     ori_img = np.zeros((350, 350), dtype='float32')
            #     ori_img[63:287, 63:287] = img
            #     scipy.misc.imsave('visual_result/messidor_05/DME_spatial_attention' + str(i) + '.jpg', ori_img)
            #
            # exit(0)
            x1 = self.avgpool(self.branch_bam1(x))
            x2 = self.avgpool(self.branch_bam2(x))

            # #  task specific feature
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x1 = self.classifier_dep1(x1)
            x2 = self.classifier_dep2(x2)

            out1 = self.classifier_specific_1(x1)
            out2 = self.classifier_specific_2(x2)
            #
            # # learn task correlation
            x1_att = self.branch_bam3(x1.view(x1.size(0), -1, 1, 1))
            x2_att = self.branch_bam4(x2.view(x2.size(0), -1, 1, 1))

            x1_att = x1_att.view(x1_att.size(0), -1)
            x2_att = x2_att.view(x2_att.size(0), -1)

            if self.choice == "both":
                x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
                x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)
            elif self.choice == "dr2dme":
                x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)
            elif self.choice == "dme2dr":
                x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
            else:
                x1 = x1 
                x2 = x2

            # for i in range(0,1):
            #     img = resize(scale[8, i, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #              cval=0, clip=True, preserve_range=True)
            #     ori_img = np.zeros((350, 350), dtype='float32')
            #     ori_img[63:287, 63:287] = img
            #     scipy.misc.imsave('visual_result/08/spatial_attention' + str(i) + '.jpg', ori_img)
            # for i in range(0,1):
            #     img = resize(scale[9, i, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #              cval=0, clip=True, preserve_range=True)
            #     ori_img = np.zeros((350, 350), dtype='float32')
            #     ori_img[63:287, 63:287] = img
            #     scipy.misc.imsave('visual_result/09/spatial_attention' + str(i) + '.jpg', ori_img)
            # for i in range(0,1):
            #     img = resize(scale[16, i, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
            #              cval=0, clip=True, preserve_range=True)
            #     ori_img = np.zeros((350, 350), dtype='float32')
            #     ori_img[63:287, 63:287] = img
            #     scipy.misc.imsave('visual_result/16/spatial_attention' + str(i) + '.jpg', ori_img)

            # final classifier
            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)

            return x1, x2, out1, out2

        if self.crosspatialCBAM:
            x = self.dropout(x)
            x1 = self.branch_bam1(x)
            x2 = self.branch_bam2(x)
            # print (x1.shape)
            # print (x2.shape)
            out1 = self.avgpool(x1)
            out2 = self.avgpool(x2)
            out1 = out1.view(out1.size(0), -1)
            out2 = out2.view(out2.size(0), -1)
            out1 = self.classifier_specific_1(out1)
            out2 = self.classifier_specific_2(out2)
            
            x1_att = self.branch_bam3(x1)
            x2_att = self.branch_bam4(x1)

            # print (x1.shape, x2_att.shape)
            # element-wise sum
            x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
            x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)

            x1 = self.avgpool(x1)
            x2 = self.avgpool(x2)
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            # x1 = self.relu_final(self.bn_final1(self.conv_final1(x1)))
            # x2 = self.relu_final(self.bn_final1(self.conv_final1(x2)))

            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)

            return x1, x2, out1, out2


        if self.CAN_TS:
            x = self.dropout(x)
            # print (x.shape)
            # print (self.branch_bam1(x).shape)
            # exit(0)
            x1 = self.avgpool(self.branch_bam1(x))
            x2 = self.avgpool(self.branch_bam2(x))
            # x1 = torch.cat((x1, x1_dr, x2_dr, x3_dr), 1)
            # x2 = torch.cat((x2, x1_dme, x2_dme, x3_dme), 1)
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            out1 = self.classifier1(x1)
            out2 = self.classifier2(x2)
            return out1, out2


        x = self.avgpool(x)
        out = x.view(x.size(0), -1)
        if self.chen:
            out = self.fc_out(out)
            out = self.dropout(out)
            out1 = self.classifier1_3(self.dropout(self.classifier1_2(self.dropout(self.classifier1_1(out)))))
            out2 = self.classifier2_3(self.dropout(self.classifier2_2(self.dropout(self.classifier2_1(out)))))
            return out1, out2
        if self.liu:
            out = self.dropout(out)
            out2 = self.branch2(out)
            out2 = self.dropout(out2)
            out1 = torch.cat((out, out2), dim=1)
            out1 = self.classifier1(out1)
            out2 = self.classifier2(out2)
            return out1, out2
        elif self.multitask:
            out1 = self.classifier1(out)
            out2 = self.classifier2(out)
            return out1, out2
        else:
            if self.num_classes == 2:
                out = self.classifier1(out)
            elif self.num_classes == 3:
                out = self.classifier2(out)
            elif self.num_classes == 5:
                out = self.classifier3(out)
            return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def image_cv(x1):
    import cv2
    filename = ["IDRiD_018", 'IDRiD_031', 'IDRiD_076', 'IDRiD_002', 'IDRiD_009', 'IDRiD_021']
    for num in range(len(filename)):
        for i in range(0, 30):
            raw_img = cv2.imread("visual_result/" + filename[num] + "/" + filename[num] + ".jpg")
            img = resize(x1[num, i, :, :].cpu().data.numpy(), (224, 224), order=3, mode='constant',
                         cval=0, clip=True, preserve_range=True)
            ori_img = np.zeros((350, 350), dtype='float32')
            ori_img[63:287, 63:287] = img
            color_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)
            color_img = resize(color_img, (raw_img.shape[0], raw_img.shape[1]), order=3, mode='constant',
                               cval=0, clip=True, preserve_range=True)
            scipy.misc.toimage(color_img, cmin=0.0, cmax=1.0).save(
                'visual_result/' + filename[num] + '/DME_attention' + str(i) + '.jpg')
