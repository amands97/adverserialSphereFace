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


featureNet = getattr(net_sphere,args.net)()
if args.model_folder == -1:
    featureNet.load_state_dict(torch.load('saved_models_ce_masked/featureNet_' + args.epoch_num + '.pth'))
else:
    featureNet.load_state_dict(torch.load('saved_models_ce_masked{}/featureNet_'.format(args.model_folder) + args.epoch_num + '.pth'))

# featureNet.cuda()
featureNet.eval()


fcNet = getattr(net_sphere, "fclayers")()
if args.model_folder == -1:
    fcNet.load_state_dict(torch.load("saved_models_ce_masked/fcNet_"+ args.epoch_num + ".pth"))
else:
    fcNet.load_state_dict(torch.load("saved_models_ce_masked{}/fcNet_".format(args.model_folder)+ args.epoch_num + ".pth"))

# fcNet.cuda()
fcNet.feature = True
fcNet.eval()

# net = getattr(net_sphere,args.net)()
# net.load_state_dict(torch.load(args.model))
# net.cuda()
# net.eval()
# net.feature = True

zfile = zipfile.ZipFile(args.lfw)
# print(zfile)
# landmark = {}
# with open('data/lfw_landmark.txt') as f:
#     landmark_lines = f.readlines()
# for line in landmark_lines:
#     l = line.replace('\n','').split('\t')
#     landmark[l[0]] = [int(k) for k in l[1:]]

with open('data/ardata_pairs.txt') as f:
    pairs_lines = f.readlines()

pairs_lines = pairs_lines
count1 = 0
count2 = 0
for i in range(len(pairs_lines)):
    if (i%100 == 0):
        print("done:", i)
    p = pairs_lines[i].replace('\n','').split(" ")

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+'{:02}.bmp'.format(int(p[1]))
        name2 = p[0]+'/'+'{:02}.bmp'.format(int(p[2]))
        count1 += 1
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+'{:02}.bmp'.format(int(p[1]))
        name2 = p[2]+'/'+'{:02}.bmp'.format(int(p[3]))
        count2 += 1
    # print(name1, name2)
    # pass
    img1 = cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    img2 = cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    img1 = cv2.resize(img1, (96, 112))
    img2 = cv2.resize(img2, (96, 112))
    # img1.show()
    # print(img1.shape)
    # cv2.imshow('ImageWindow', img1)
    # cv2.waitKey()
    # continue


    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    img = Variable(torch.from_numpy(img).float(),volatile=True)#.cuda()

    # print(img.shape)
    # from matplotlib import pyplot as  plt
    # input_image = img[:1]
    # print(input_image.shape)
    # output = input_image.data
    # print(output.size(1))
    # fig, axarr = plt.subplots(output.size(1))
    # print(output.size())
    # for idx in range(output.size(1)):
    #     # print(output[:, idx].shape)
    #     # print(output.shape)
    #     # print(output[idx].shape)
    #     axarr[idx].imshow(output[0, idx])
    # plt.show()


    # import sys
    # sys.exit()








    # output = net(img)
    output = featureNet(img)
    # print(output)
    output = fcNet(output)
    # print(output)
    f = output.data
    f1,f2 = f[0],f[2]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    print(cosdistance)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=len(pairs_lines), n_folds=10, shuffle=True)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in enumerate(folds):
    
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print(accuracy)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
