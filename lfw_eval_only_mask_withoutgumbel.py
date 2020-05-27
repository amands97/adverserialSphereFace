# CUDA_VISIBLE_DEVICES=1 python lfw_eval.py --lfw lfw.zip --epoch_num 2
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
from torchvision import transforms
import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from gumbel import gumbel_softmax
import adversary_full as adversary
def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n/n_folds:(i+1)*n/n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold



parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='lfw/lfw.zip', type=str)
parser.add_argument('--model','-m', default='sphere20a.pth', type=str)
parser.add_argument('--epoch_num', type=str)
parser.add_argument('--model_folder', type=int, default = -1)

args = parser.parse_args()

predicts=[]
use_cuda = torch.cuda.is_available()



featureNet = getattr(net_sphere,args.net)()
# featureNet.load_state_dict(torch.load('saved_models_ce/featureNet_' + args.epoch_num + '.pth'))
# featureNet.cuda()

# we dont need maskNet here right?
maskNet = getattr(adversary, "MaskMan")()
if args.model_folder == -1:
    maskNet.load_state_dict(torch.load('saved_models_ce_masked/maskNet_' + args.epoch_num + '.pth'))
else:
    maskNet.load_state_dict(torch.load('saved_models_ce_masked{}/maskNet_'.format(args.model_folder) + args.epoch_num + '.pth'))
# maskNet.load_state_dict(torch.load("saved_models_ce_masked19/maskNet_" + args.epoch_num +".pth"))
# maskNet.cuda()
maskNet.eval()

fcNet = getattr(net_sphere, "fclayers")()
# fcNet.load_state_dict(torch.load("saved_models_ce/fcNet_"+ args.epoch_num + ".pth"))
# fcNet.cuda()
fcNet.feature = True
fcNet.eval()

if use_cuda:
    featureNet.cuda()
    fcNet.cuda()
    maskNet.cuda()
else:
    featureNet.cpu()
    fcNet.cpu()
    maskNet.cpu()
# net = getattr(net_sphere,args.net)()
# net.load_state_dict(torch.load(args.model))
# net.cuda()
# net.eval()
# net.feature = True

zfile = zipfile.ZipFile(args.lfw)
# print(zfile)
landmark = {}
with open('data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

for i in range(6000):
    if (i%100 == 0):
        print("done:", i)
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
    print(name1)
    print(name2)
    # img1 = cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    # img2 = cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    # img1 = cv2.resize(img1, (96, 112))
    # img2 = cv2.resize(img2, (96, 112))
    img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
    img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])
    print("shadnoasjdnajosd", img1.shape)
    # print(img1)
    # from matplotlib import pyplot as plt
    # plt.imshow(img1)
    cv2.imwrite("1.jpg", img1)
    cv2.imwrite("2.jpg", img2) 
    print(img1.shape)
    
    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    if use_cuda:
        img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    else:
        img = Variable(torch.from_numpy(img).float(),volatile=True)

    # output = net(img)
    output = featureNet(img)
    # print(output)
    mask = maskNet(img)
    # print(mask)
    # mask = gumbel_softmax(mask, tau = 0.001)
    # mask = nn.Upsample(scale_factor = 16, mode = 'nearest')(mask)
    # print(mask)
    # print(mask.shape)
    mask = ((mask > 0).type(torch.uint8))
    # mask = 2 * mask - 1
    # print(mask == 0)
    # a_set = torch.unique((torch.Tensor([i for i in range(256)])))
    outimg = img.numpy() * mask.detach().numpy()
    import random
    # print((outimg == 0).shape)
    outimg[outimg == 0] = [random.uniform(-1,1 ) for _ in range(len(outimg[outimg == 0]))]
    # print(outimg)
    # print(img[:, 0, :][mask == 0])
    # img[img < 0] 
    # print(img[img == 0].size())
    # print(img[img == -1].size())
    # import sys
    # sys.exit()
    # import sys
    # sys.exit()
    # print(img[0,0])
    # outimg = img.numpy() * mask.detach().numpy()
    # outimg = outimg[:, 0, :, :].unsqueeze(1)
    # print(outimg.shape)
    # print(outimg[0, 0])
    # outimg[:, :].permute(2, 1, 0)
    # print(outimg.shape)
    # print(outimg.permute())
    # print(outimg.unsqueeze(1).shape)

    for i in range(4):
        # print(mask[i, 0])
        # print(outimg[i])
        # image = cv2.resize(outimg[i].detach().unsqueeze(0).numpy(), (96, 112), interpolation = cv2.INTER_AREA)
        print(outimg[i].shape)
        print(cv2.cvtColor(outimg[i].transpose(1, 2, 0), cv2.COLOR_BGR2RGB).shape)
        print("asdasd")
        image = outimg[i].transpose(1,2,0)
        # image = cv2.cvtColor(outimg[i].transpose(1,2,0), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (96, 112))
        image = image * 128.0 + 127.5
        # image = image.transpose((2, 0, 1))
        print(image.shape)
        # print(outimg[])
        cv2.imwrite("face" + str(i)+  ".jpg", image)

        # image.save("face" + str(i)+  ".jpg")

        image = transforms.ToPILImage(mode='L')(mask[i])
        image = image.resize((96, 112))
        image.save("mask" + str(i)+  ".jpg")
    # import sys
    # sys.exit()
    # output = fcNet(output)
    # # print(output)
    # f = output.data
    # f1,f2 = f[0],f[2]
    # cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    # predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))
    import time
    time.sleep(5)
    # if i == 0:
    import sys
    sys.exit()

accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in enumerate(folds):
    
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
