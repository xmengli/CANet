import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import random
import numpy as np
import argparse
import os
import shutil
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('dataset', metavar='Dataset')
parser.add_argument('model_dir', metavar='savedir')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--decay_epoch', default=15, type=int)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument("--invalid", action="store_true")


parser.add_argument("--multitask", action="store_true")
parser.add_argument("--liu", action="store_true")
parser.add_argument("--chen", action="store_true")
parser.add_argument("--crossCBAM", action="store_true")


parser.add_argument("--net_type", default="regular", type=str)
parser.add_argument("--channels", default=109, type=int)
parser.add_argument("--nodes", default=32, type=int)
parser.add_argument("--graph_model", default="WS", type=str)
parser.add_argument("--K", default=4, type=int)
parser.add_argument("--P", default=0.75, type=float)



# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--base_lr", default=3e-2, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument('--momentum', default=5e-5, type=float, metavar='M',
                    help='momentum')


best_acc1 = 0
minimum_loss = 1.0
count = 0
test_times = [350, 351, 349]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main():
    args = parser.parse_args()

    my_whole_seed = args.seed

    torch.manual_seed(my_whole_seed)
    torch.cuda.manual_seed_all(my_whole_seed)
    torch.cuda.manual_seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    random.seed(my_whole_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    #
    # cudnn.deterministic = False
    # cudnn.benchmark = True


    main_worker(args.gpu, args)

def worker_init_fn(worker_id):
    random.seed(1 + worker_id)

def main_worker(gpu, args):
    global best_acc1
    global minimum_loss
    global count
    args.gpu = gpu

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    if args.arch == "vgg11":
        from models.vgg import vgg11
        model = vgg11(num_classes=args.num_class, crossCBAM=args.crossCBAM)
    elif args.arch == "resnet50":
        from models.resnet50 import resnet50
        model = resnet50(num_classes=args.num_class, multitask=args.multitask, liu=args.liu,
                 chen=args.chen, flagCBAM=False, crossCBAM=args.crossCBAM)
    elif args.arch == "resnet34":
        from models.resnet50 import resnet34
        model = resnet34(num_classes=args.num_class, multitask=args.multitask, liu=args.liu,
                 chen=args.chen, flagCBAM=False, crossCBAM=args.crossCBAM)
    elif args.arch == "resnet18":
        from models.resnet50 import resnet18
        model = resnet18(num_classes=args.num_class, multitask=args.multitask, liu=args.liu,
                 chen=args.chen, flagCBAM=False, crossCBAM=args.crossCBAM)
    elif args.arch == "densenet161":
        from models.densenet import densenet161
        model = densenet161(num_classes=args.num_class, multitask=args.multitask, cosface=False, liu=args.liu,
                    chen=args.chen)
    elif args.arch == "wired":
        from models.wirednetwork import CNN
        model = CNN(args, num_classes=args.num_class)
    else:
        print ("no backbone model")

    if args.pretrained:
        print ("==> Load pretrained model")
        model_dict = model.state_dict()
        pretrain_path = {"resnet50": "pretrain/resnet50-19c8e357.pth",
                         "resnet34": "pretrain/resnet34-333f7ec4.pth",
                         "resnet18": "pretrain/resnet18-5c106cde.pth",
                         "densenet161": "pretrain/densenet161-8d451a50.pth",
                         "vgg11": "pretrain/vgg11-bbd30ac9.pth"}[args.arch]
        pretrained_dict = torch.load(pretrain_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict.pop('classifier.weight', None)
        pretrained_dict.pop('classifier.bias', None)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,  map_location={'cuda:4':'cuda:0'})
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(0)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    size  = 224
    # tra = transforms.Compose([
    #             # transforms.Resize(256),
    #             transforms.RandomResizedCrop(size),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomVerticalFlip(),
    #             # transforms.RandomRotation(90),
    #             # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    # tra_test = transforms.Compose([
    #         transforms.Resize(size+32),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor(),
    #         normalize])

    tra = transforms.Compose([
        transforms.Resize(350),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
        transforms.RandomCrop(224),
        #            transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # tra = transforms.Compose([
    #     transforms.Resize(350),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #     transforms.ToTensor(),
    #     normalize])

    #
    tra_test = transforms.Compose([
        transforms.Resize(350),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    #
    # from autoaugment import ImageNetPolicy
    # tra =transforms.Compose([
    #      transforms.RandomResizedCrop(224),
    #      transforms.RandomHorizontalFlip(),
    #      ImageNetPolicy(),
    #      transforms.ToTensor(),
    #      normalize])

    # image = PIL.Image.open(path)
    # policy = ImageNetPolicy()
    # transformed = policy(image)


    if args.dataset == 'amd':
        from datasets.amd_dataset import traindataset
    elif args.dataset == 'pm':
        from datasets.pm_dataset import traindataset
    elif args.dataset == "drdme":
        from datasets.drdme_dataset import traindataset
    elif args.dataset == "missidor":
        from datasets.missidor import traindataset
    else:
        print ("no dataset")
        exit(0)

    if args.evaluate:
        # result = validate(val_loader, model, args)
        result = multi_validate(model, test_times, normalize, traindataset, args)
        print ("acc_dr, acc_dme, acc_joint", result)
        return



    val_dataset = traindataset(root=args.data, mode = 'val', transform=tra_test, num_class=args.num_class,
                               multitask=args.multitask)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)



    train_dataset = traindataset(root=args.data, mode='train', transform=tra, num_class=args.num_class,
                                 multitask=args.multitask)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,worker_init_fn=worker_init_fn)

    writer = SummaryWriter()
    writer.add_text('Text', str(args))

    # from lr_scheduler import LRScheduler
    # lr_scheduler = LRScheduler(optimizer, len(train_loader), args)

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        lr = adjust_learning_rate(optimizer, epoch, args)
        writer.add_scalar("lr", lr, epoch)
        # train for one epoch
        loss_train = train(train_loader, model, criterion, optimizer, args)
        writer.add_scalar('Train loss', loss_train, epoch)

        # evaluate on validation set
        if epoch % 20 == 0:
            acc_dr, acc_dme, joint_acc = validate(val_loader, model, args)
            writer.add_scalar("Val acc_dr", acc_dr, epoch)
            writer.add_scalar("Val acc_dme", acc_dme, epoch)
            writer.add_scalar("Val acc_joint", joint_acc, epoch)
            is_best = joint_acc >= best_acc1
            best_acc1 = max(joint_acc, best_acc1)

        if not args.invalid:
            if is_best:
                save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best, filename = "checkpoint" + str(epoch) + ".pth.tar", save_dir=args.model_dir)

def train(train_loader, model, criterion, optimizer,  args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # lr = lr_scheduler.update(i, epoch)
        # writer.add_scalar("lr", lr, epoch)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking = True)

        if args.multitask:
            target = [item.cuda(args.gpu, non_blocking = True) for item in target]
        else:
            target = target.cuda(args.gpu, non_blocking = True)


        # compute output
        output = model(input)
        if args.multitask:
            loss1 = criterion(output[0], target[0])
            loss2 = criterion(output[1], target[1])
            # if epoch < 100:
            #     loss = (loss1*0.5  + loss2*0.5)
            # else:
            #     total = []
            #     total.append(loss1)
            #     total.append(loss2)
            #     c = [-torch.mul(torch.mul(torch.mul(torch.mul(item, item), item),item),item) * \
            #          torch.log(torch.max(torch.FloatTensor([0.01]).cuda(),1 - item)) for item in total]
            #     loss = (c[0]+c[1]) /2

            loss = (loss1 * 0.5 + loss2 * 0.5)
        else:
            loss = criterion(output, target)


        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    all_target = []
    all_target_dme = []
    all_output = []
    all_name = []
    all_output_dme = []
    with torch.no_grad():
        for i, (input, target, name) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            if args.multitask:
                target = [item.cuda(args.gpu, non_blocking=True) for item in target]
            else:
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            if args.multitask:
                output0 = torch.softmax(output[0], dim=1)
                all_target.append(target[0].cpu().data.numpy())
                all_output.append(output0.cpu().data.numpy())
                output1 = torch.softmax(output[1], dim=1)
                all_target_dme.append(target[1].cpu().data.numpy())
                all_output_dme.append(output1.cpu().data.numpy())
            else:
                output = torch.softmax(output, dim=1)
                all_target.append(target.cpu().data.numpy())
                all_output.append(output.cpu().data.numpy())
            all_name.append(name)


    all_target = [item for sublist in all_target for item in sublist]
    all_output = [item for sublist in all_output for item in sublist]
    all_target_dme = [item for sublist in all_target_dme for item in sublist]
    all_output_dme = [item for sublist in all_output_dme for item in sublist]

    # acc
    acc_dr = accuracy_score(all_target, np.argmax(all_output,axis=1))
    acc_dme = accuracy_score(all_target_dme, np.argmax(all_output_dme, axis=1))

    # joint acc
    joint_result = np.vstack((np.argmax(all_output, axis=1), np.argmax(all_output_dme, axis=1)))
    joint_target = np.vstack((all_target, all_target_dme))
    joint_acc = ((np.equal(joint_result, joint_target) == True).sum(axis=0) == 2).sum()/joint_result.shape[1]

    return acc_dr, acc_dme, joint_acc

def multi_validate(model, test_times, normalize, traindataset, args):
    # switch to evaluate mode
    model.eval()

    all_output = []
    all_output_dme = []

    for times in test_times:
        tra_test = transforms.Compose([
            transforms.Resize(times),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        val_dataset = traindataset(root=args.data, mode='val', transform=tra_test, num_class=args.num_class,
                                   multitask=args.multitask)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


        all_target = []
        all_target_dme = []
        sub_output = []
        all_name = []
        sub_output_dme = []
        with torch.no_grad():
            for i, (input, target, name) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)

                if args.multitask:
                    target = [item.cuda(args.gpu, non_blocking=True) for item in target]
                else:
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(input)

                if args.multitask:
                    output0 = torch.softmax(output[0], dim=1)
                    all_target.append(target[0].cpu().data.numpy())
                    sub_output.append(output0.cpu().data.numpy())
                    output1 = torch.softmax(output[1], dim=1)
                    all_target_dme.append(target[1].cpu().data.numpy())
                    sub_output_dme.append(output1.cpu().data.numpy())
                else:
                    output = torch.softmax(output, dim=1)
                    all_target.append(target.cpu().data.numpy())
                    sub_output.append(output.cpu().data.numpy())
                all_name.append(name)


        all_target = [item for sublist in all_target for item in sublist]
        sub_output = [item for sublist in sub_output for item in sublist]
        all_target_dme = [item for sublist in all_target_dme for item in sublist]
        sub_output_dme = [item for sublist in sub_output_dme for item in sublist]


        all_output.append(sub_output)
        all_output_dme.append(sub_output_dme)


    all_output = [sum(x) for x in zip(all_output[0], all_output[1], all_output[2]
                                      )]
    all_output_dme = [sum(x) for x in zip(all_output_dme[0], all_output_dme[1], all_output_dme[2]
                                       )]


    # acc
    acc_dr = accuracy_score(all_target, np.argmax(all_output,axis=1))
    acc_dme = accuracy_score(all_target_dme, np.argmax(all_output_dme, axis=1))

    # joint acc
    joint_result = np.vstack((np.argmax(all_output, axis=1), np.argmax(all_output_dme, axis=1)))
    joint_target = np.vstack((all_target, all_target_dme))
    joint_acc = ((np.equal(joint_result, joint_target) == True).sum(axis=0) == 2).sum()/joint_result.shape[1]

    return acc_dr, acc_dme, joint_acc


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir= 'file'):

    root = save_dir + "/"
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(state, root+'model_converge.pth.tar')
    # if is_best:
    #     shutil.copyfile(root+filename, root+'model_converge.pth.tar')


def save_result2txt(savedir, all_output_dme, all_output,all_target_dme,all_target):
    np.savetxt(savedir+"/output_dme.txt", all_output_dme, fmt='%.4f')
    np.savetxt(savedir+"/output_dr.txt", all_output, fmt='%.4f')
    np.savetxt(savedir+"/target_dme.txt", all_target_dme)
    np.savetxt(savedir+"/target_dr.txt", all_target)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def multi_class_auc(all_target,all_output, num_c = None):

    all_output = np.stack(all_output)
    all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    auc_sum = []
    for num_class in range(0, num_c):
        auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
        auc_sum.append(auc)
    auc = sum(auc_sum) / float(len(auc_sum))

    return auc

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels].cuda()


if __name__ == '__main__':
    main()