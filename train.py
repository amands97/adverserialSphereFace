# CUDA_VISIBLE_DEVICES=2 python train.py --datase CASIA-WebFace.zip --bs 256


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
parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--dataset', default='../../dataset/face/casia/casia.zip', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=256, type=int, help='')
parser.add_argument('--checkpoint', default=-1, type=int, help='if use checkpoint then mention the number, otherwise training from scratch')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

writer = SummaryWriter()
n_iter = 0

def alignment(src_img,src_pts):
    of = 2
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def dataset_load(name,filename,pindex,cacheobj,zfile):
    position = filename.rfind('.zip:')
    
    zipfilename = filename[0:position+4]
    nameinzip = filename[position+5:]
    split = nameinzip.split('\t')
    nameinzip = split[0]
    classid = int(split[1])
    src_pts = []
    for i in range(5):
        src_pts.append([int(split[2*i+2]),int(split[2*i+3])])

    data = np.frombuffer(zfile.read(nameinzip),np.uint8)
    img = cv2.imdecode(data,1)
    img = alignment(img,src_pts)

    if ':train' in name:
        if random.random()>0.5: img = cv2.flip(img,1)
        if random.random()>0.5:
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img = img[ry:ry+112,rx:rx+96,:]
        else:
            img = img[2:2+112,2:2+96,:]
    else:
        img = img[2:2+112,2:2+96,:]


    img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    img = ( img - 127.5 ) / 128.0
    label = np.zeros((1,1),np.float32)
    label[0,0] = classid
    return (img,label)


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


def getKernel():
    # https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176/2
    kernel = torch.ones((3,3))
    kernel[1, 1] = -8
    kernel = kernel/-8
    # b, c, h, w = x.shape (hard coded here)
    b, c, h, w = (1,1, 7, 6)
    kernel = kernel.type(torch.FloatTensor)
    kernel = kernel.repeat(c, 1, 1, 1)
    return kernel

def train(epoch,args):
    featureNet.train()
    maskNet.train()
    fcNet.train()
    train_loss = 0
    classification_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    ds = ImageDataset(args.dataset,dataset_load,'data/casia_landmark.txt',name=args.net+':train',
        bs=args.bs,shuffle=True,nthread=6,imagesize=128)
    global n_iter
    while True:
        n_iter += 1
        img,label = ds.get()
        if img is None: break
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:,0]).long()
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        optimizerMask.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        features = featureNet(inputs)
        mask = maskNet(features)
        mask = gumbel_softmax(mask)
        # print(mask.size())
        maskedFeatures = torch.mul(mask, features)
        # print(features.shape, mask.shape, maskedFeatures.shape)

        outputs = fcNet(maskedFeatures)
        outputs1 = outputs[0] # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        # training the advNet:
        lossAdv = criterion(outputs, targets)
        # print(conv2d(mask, laplacianKernel, stride=1, groups=1).size())
        lossCompact = torch.sum(conv2d(mask, laplacianKernel, stride=1, groups=1))
        # lossSize   #L1 norm of the mask to make the mask sparse.
        if use_cuda:
            lossSize = F.l1_loss(mask, target=torch.ones(mask.size()).cuda(), size_average = False)
        else:
            lossSize = F.l1_loss(mask, target=torch.ones(mask.size()), size_average = False)
        # print("advnet:", - criterion2(outputs1, targets).data/10, lossCompact.data/1000000, lossSize.data/10000)
        writer.add_scalar('Loss/adv-classification', - criterion2(outputs1, targets)/100 , n_iter)
        writer.add_scalar('Loss/adv-compactness', lossCompact/1000000, n_iter)
        writer.add_scalar('Loss/adv-size', lossSize/10000, n_iter)
        loss = - criterion2(outputs1, targets)/100 + lossCompact/1000000 + lossSize/10000
        writer.add_scalar('Accuracy/adv-totalLoss', loss, n_iter)
        lossd = loss.data
        loss.backward(retain_graph=True)
        optimizerMask.step()
        
        optimizerFC.zero_grad()
        lossC = criterion(outputs, targets)
        lossClassification = lossC.data
        lossC.backward()
        optimizerFC.step()
        classification_loss += lossClassification
        train_loss += loss.data

        # print("classification loss:", classification_loss / (batch_idx + 1))
        writer.add_scalar('Loss/classn-loss', classification_loss/(batch_idx + 1), n_iter)
        writer.add_scalar('Loss/adv-avgloss', train_loss/(batch_idx + 1), n_iter)
        # printoneline(dt(),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f %.2f %d\n'
            # % (epoch,train_loss/(batch_idx+1), 100.0*correct/total, correct, total, 
            # lossd, criterion.lamb, criterion.it))
        writer.add_scalar('Accuracy/classification', 100* correct/total, n_iter)
        # writer.add_scalar
        writer.add_scalar('Accuracy/correct', correct, n_iter)
        batch_idx += 1
        # break
    print('')


if args.checkpoint == -1:
    featureNet = getattr(net_sphere,args.net)()

    featureNet.load_state_dict(torch.load('model/sphere20a_20171020.pth'))

    maskNet = getattr(adversary, "MaskMan")(512)
    fcNet = getattr(net_sphere, "fclayers")()
    pretrainedDict = torch.load('model/sphere20a_20171020.pth')
    fcDict = {k: pretrainedDict[k] for k in pretrainedDict if k in fcNet.state_dict()}
    fcNet.load_state_dict(fcDict)
    laplacianKernel = getKernel()
else:
    featureNet = getattr(net_sphere,args.net)()
    featureNet.load_state_dict(torch.load('saved_models_ce/featureNet_' + str(args.checkpoint) + '.pth'))

    maskNet = getattr(adversary, "MaskMan")(512)
    maskNet.load_state_dict(torch.load('saved_models_ce/maskNet_' + str(args.checkpoint) + '.pth'))
    fcNet = getattr(net_sphere, "fclayers")()
    # pretrainedDict = torch.load('model/sphere20a_20171020.pth')
    # fcDict = {k: pretrainedDict[k] for k in pretrainedDict if k in fcNet.state_dict()}
    fcNet.load_state_dict(torch.load('saved_models_ce/fcNet_'+ str(args.checkpoint)+ '.pth'))
    laplacianKernel = getKernel()
# print(advNet)
# net = getattr(net_sphere, "newNetwork")(net1, advNet)
if use_cuda:
    featureNet.cuda()
    maskNet.cuda()
    fcNet.cuda()
    laplacianKernel =  laplacianKernel.cuda()

criterion = net_sphere.AngleLoss()
optimizerFC = optim.SGD(list(featureNet.parameters()) + list(fcNet.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # optimizerFeature = optim.SGD(featureNet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizerMask = optim.SGD(maskNet.parameters(), lr = args.lr/1000, momentum=0.9, weight_decay=5e-4)
criterion2 = torch.nn.CrossEntropyLoss()

print('start: time={}'.format(dt()))
for epoch in range(0, 50):
    if epoch in [0,10,15,18]:
        if epoch!=0: args.lr *= 0.1
        optimizerFC = optim.SGD(list(featureNet.parameters()) + list(fcNet.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # optimizerFeature = optim.SGD(featureNet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizerMask = optim.SGD(maskNet.parameters(), lr = args.lr/1000, momentum=0.9, weight_decay=5e-4)
    if args.checkpoint >= epoch:
        continue
        # optimizerFC = optim.SGD(fcNet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train(epoch,args)
    save_model(featureNet, 'saved_models_ce/featureNet_{}.pth'.format(epoch))
    save_model(maskNet, 'saved_models_ce/maskNet_{}.pth'.format(epoch))
    save_model(fcNet, 'saved_models_ce/fcNet_{}.pth'.format(epoch))

print('finish: time={}\n'.format(dt()))

