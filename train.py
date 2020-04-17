# CUDA_VISIBLE_DEVICES=2 python train.py --datase CASIA-WebFace.zip --bs 256


# python train.py --dataset casia.zip --bs 10 --lr 0.0001 --lrfc 0.00005 --mom 0.0000 --momfc 0.0000 --checkpoint blok
#  fine lr 0.01 and lrfc 0.0000001 use mom 0. maybe try 0.001

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys
import cv2
import random,datetime
import argparse
import numpy as np
np.warnings.filterwarnings('ignore')
from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
import adversary
from gumbel import gumbel_softmax
from torch.nn.functional import conv2d # for the kernel
from torch.utils.tensorboard import SummaryWriter
from aux import *
# to import all the necessary changes
parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--dataset', default='../../dataset/face/casia/casia.zip', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrfc', default=0.1, type=float, help='learning rate classifier')

parser.add_argument('--bs', default=256, type=int, help='')
parser.add_argument('--mom', default=0.9, type=float, help='momentum')
parser.add_argument('--momfc', default=0.9, type=float, help='momentum classifier')

parser.add_argument('--checkpoint', default=-1, type=int, help='if use checkpoint then mention the number, otherwise training from scratch')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

writer = SummaryWriter()
n_iter = 0

def train(epoch,args):
    featureNet.train()
    maskNet.train()
    fcNet.train()
    train_loss = 0
    classification_loss = 0
    correct = 0
    total = 0
    correct2 = 0
    total2 = 0
    batch_idx = 0
    ds = ImageDataset(args.dataset,dataset_load,'data/casia_landmark.txt',name=args.net+':train',
        bs=args.bs,shuffle=True,nthread=6,imagesize=128)

    global n_iter
    while True:
        if batch_idx % 50 == 0 and batch_idx > 0:
            print(batch_idx)

        n_iter += 1
        img,label = ds.get()
        if img is None: break
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:,0]).long()
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        # outputs = fcNet(featureNet(inputs))
        # outputs1 = outputs[0] # 0=cos_theta 1=phi_theta
        # _, predicted = torch.max(outputs1.data, 1)
        # total2 += targets.size(0)
        # if use_cuda:
        #     correct2 += predicted.eq(targets.data).cpu().sum()
        # else:
        #     correct2 += predicted.eq(targets.data).sum()
        # writer.add_scalar("Accuracy/true", 100 * (correct2)/(total2 * 1.0), n_iter)

        optimizerMask.zero_grad()
        # optimizerFC.zero_grad()
        mask =gumbel_softmax(maskNet(inputs))
        mask = upsampler(mask)
        maskedFeatures = torch.mul(mask, inputs)
        outputs = fcNet(featureNet(maskedFeatures))
        outputs1 = outputs[0] # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        if use_cuda:
            correct += predicted.eq(targets.data).cpu().sum()
        else:
            correct += predicted.eq(targets.data).sum()
        lossAdv = criterion(outputs, targets)
        # lossCompact = torch.sum(conv2d(mask, laplacianKernel, stride=1, groups=1))
        if use_cuda:
            lossSize1 = F.l1_loss(mask, target=torch.ones(mask.size()).cuda(), reduction = 'mean')
        else:
            lossSize1 = F.l1_loss(mask, target=torch.ones(mask.size()), reduction = 'mean')
        lossSize = 0
        if lossSize1 > 0.25:
            lossSize = (100*(lossSize1 - 0.25)).pow(2)
        elif lossSize1 < 0.10:
            lossSize = 10000*(100 * (0.10 - lossSize1).pow(2)) 
        writer.add_scalar('Loss/adv-classification', -lossAdv/10, n_iter)
        # writer.add_scalar('Loss/adv-compactness', lossCompact/10, n_iter)
        writer.add_scalar('Loss/adv-size', lossSize, n_iter)
        loss = (-lossAdv)/100000000  + lossSize
        writer.add_scalar('Accuracy/adv-totalLoss', loss, n_iter)
        lossd = loss.data
        loss.backward()
        optimizerMask.step()

        # set this optimizer mask grad to be zero again
        # optimizerMask.zero_grad()
        optimizerFC.zero_grad()

        mask = gumbel_softmax(maskNet(inputs))
        mask = upsampler(mask)
        maskedFeatures = torch.mul(mask, inputs)
        maskedFeatures = inputs
        outputs = fcNet(featureNet(maskedFeatures))
        total += targets.size(0)


        lossC = criterion(outputs, targets)
        lossClassification = lossC.data
        lossC.backward()
        optimizerFC.step()
        classification_loss += lossClassification
        # train_loss += loss.data

        writer.add_scalar('Loss/classn-loss', classification_loss/(batch_idx + 1), n_iter)
        # writer.add_scalar('Loss/adv-avgloss', train_loss/(batch_idx + 1), n_iter)
        writer.add_scalar('Accuracy/classification', 100* correct/(total*1.0), n_iter)
        writer.add_scalar('Accuracy/correct', correct, n_iter)
        

        batch_idx += 1
    print('')


if args.checkpoint == -1:
    featureNet = getattr(net_sphere,args.net)()

    # featureNet.load_state_dict(torch.load('model/sphere20a_20171020.pth'))

    maskNet = getattr(adversary, "MaskMan")()

    fcNet = getattr(net_sphere, "fclayers")()
    # pretrainedDict = torch.load('model/sphere20a_20171020.pth')
    # fcDict = {k: pretrainedDict[k] for k in pretrainedDict if k in fcNet.state_dict()}
    # fcNet.load_state_dict(fcDict)
    laplacianKernel = getKernel()
else:
    featureNet = getattr(net_sphere,args.net)()
    featureNet.load_state_dict(torch.load('saved_models_ce_masked/featureNet_' + str(args.checkpoint) + '.pth'))

    maskNet = getattr(adversary, "MaskMan")()
    maskNet.load_state_dict(torch.load('saved_models_ce_masked/maskNet_' + str(args.checkpoint) + '.pth'))
    fcNet = getattr(net_sphere, "fclayers")()
    # pretrainedDict = torch.load('model/sphere20a_20171020.pth')
    # fcDict = {k: pretrainedDict[k] for k in pretrainedDict if k in fcNet.state_dict()}
    fcNet.load_state_dict(torch.load('saved_models_ce_masked/fcNet_'+ str(args.checkpoint)+ '.pth'))
    laplacianKernel = getKernel()
# print(advNet)
# net = getattr(net_sphere, "newNetwork")(net1, advNet)
if use_cuda:
    featureNet.cuda()
    maskNet.cuda()
    fcNet.cuda()
    laplacianKernel =  laplacianKernel.cuda()

criterion = net_sphere.AngleLoss()
optimizerFC = optim.SGD(list(featureNet.parameters()) + list(fcNet.parameters()), lr=args.lrfc, momentum=args.momfc, weight_decay=5e-4)
optimizerMask = optim.SGD(maskNet.parameters(), lr = args.lr, momentum=args.mom,  weight_decay=5e-4)

# optimizerFC = optim.Adam(list(featureNet.parameters()) + list(fcNet.parameters()), lr=args.lrfc)
# optimizerMask = optim.Adam(maskNet.parameters(), lr = args.lr)


criterion2 = torch.nn.CrossEntropyLoss()
upsampler = torch.nn.Upsample(scale_factor = 16, mode = 'nearest')
print('start: time={}'.format(dt()))
for epoch in range(0, 100):
    if epoch in [0,10,15, 18, 30, 45, 60]:
        if epoch!=0:
            args.lr *= 0.1
            args.lrfc *= 0.1
            optimizerFC = optim.SGD(list(featureNet.parameters()) + list(fcNet.parameters()), lr=args.lrfc, momentum=args.momfc, weight_decay=5e-4)
            optimizerMask = optim.SGD(maskNet.parameters(), lr = args.lr, momentum=args.mom, weight_decay=5e-4)
        # python train.py --dataset CASIA-WebFace.zip --bs 100 --lr 0.0003  --mom 0.09 --lrfc 0.00005 --momfc 0.09 --checkpoint=10 
        # slowed the lr even more
        # optimizerFC = optim.Adam(list(featureNet.parameters()) + list(fcNet.parameters()), lr=args.lrfc)
        # optimizerMask = optim.Adam(maskNet.parameters(), lr = args.lr)


    if args.checkpoint >= epoch:
        continue
        # optimizerFC = optim.SGD(fcNet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train(epoch,args)
    save_model(featureNet, 'saved_models_ce_masked/featureNet_{}.pth'.format(epoch))
    save_model(maskNet, 'saved_models_ce_masked/maskNet_{}.pth'.format(epoch))
    save_model(fcNet, 'saved_models_ce_masked/fcNet_{}.pth'.format(epoch))

print('finish: time={}\n'.format(dt()))

