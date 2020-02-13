# CUDA_VISIBLE_DEVICES=1 python lfw_eval.py --lfw lfw.zip --epoch_num 2
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from gumbel import gumbel_softmax
import adversary

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



parser = argparse.ArgumentParser(description='PyTorch sphereface disguised dataset')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--data', default='data/FaceDisguiseDatabase.zip', type=str)
# parser.add_argument('--model','-m', default='sphere20a.pth', type=str)
parser.add_argument('--epoch_num', type=str)
args = parser.parse_args()

predicts=[]
use_cuda = torch.cuda.is_available()


if args.epoch_num == "-1":
    featureNet = getattr(net_sphere,args.net)()

    featureNet.load_state_dict(torch.load('model/sphere20a_20171020.pth'))

    # maskNet = getattr(adversary, "MaskMan")(512)
    # maskNet = getattr(adversary, "MaskMan")()

    fcNet = getattr(net_sphere, "fclayers")()
    pretrainedDict = torch.load('model/sphere20a_20171020.pth')
    fcDict = {k: pretrainedDict[k] for k in pretrainedDict if k in fcNet.state_dict()}
    fcNet.load_state_dict(fcDict)
    fcNet.feature = True


else:
    featureNet = getattr(net_sphere,args.net)()
    featureNet.load_state_dict(torch.load('saved_models_ce/featureNet_' + args.epoch_num + '.pth'))
    # featureNet.cuda()
    featureNet.eval()

    # we dont need maskNet here right?
    # maskNet = getattr(adversary, "MaskMan")()
    # maskNet.load_state_dict(torch.load("saved_models_ce/maskNet_" + args.epoch_num +".pth"))
    # maskNet.cuda()
    # maskNet.eval()

    fcNet = getattr(net_sphere, "fclayers")()
    fcNet.load_state_dict(torch.load("saved_models_ce/fcNet_"+ args.epoch_num + ".pth"))
    # fcNet.cuda()
    fcNet.feature = True
    fcNet.eval()

if use_cuda:
    featureNet.cuda()
    fcNet.cuda()
    # maskNet.cuda()
else:
    featureNet.cpu()
    fcNet.cpu()
    # maskNet.cpu()
# net = getattr(net_sphere,args.net)()
# net.load_state_dict(torch.load(args.model))
# net.cuda()
# net.eval()
# net.feature = True

zfile = zipfile.ZipFile(args.data)
# print(zfile)
landmark = {}
# with open('data/lfw_landmark.txt') as f:
#     landmark_lines = f.readlines()
# for line in landmark_lines:
#     l = line.replace('\n','').split('\t')
#     landmark[l[0]] = [int(k) for k in l[1:]]

with open('pairs.txt') as f:
    pairs_lines = f.readlines()
n = 820
for i in range(n):
    if (i%2 == 0):
        print("done:", i)
    p = pairs_lines[i].replace('\n','').split(', ')
    print(p)
    name1 = "FaceDisguiseDatabase/FaceAll_cropped/" + p[0]
    name2 = "FaceDisguiseDatabase/FaceAll_cropped/" + p[1]
    sameflag = p[2]

    # if 3==len(p):
    #     sameflag = 1
    #     name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
    #     name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    # if 4==len(p):
    #     sameflag = 0
    #     name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
    #     name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

    img1 = cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    img2 = cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    img1 = cv2.resize(img1, (96, 112))
    img2 = cv2.resize(img2, (96, 112))

    # print(img1)
    # from matplotlib import pyplot as plt
    # plt.imshow(img1)
    # import sys
    # sys.exit()
    
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

    output = fcNet(output)
    f = output.data
    f1,f2 = f[0],f[2]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=n, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in enumerate(folds):
    
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
