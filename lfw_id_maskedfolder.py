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


featureNet = getattr(net_sphere,args.net)()
if args.model_folder == -1:
    featureNet.load_state_dict(torch.load('saved_models_ce_masked/featureNet_' + args.epoch_num + '.pth'))
else:
    featureNet.load_state_dict(torch.load('saved_models_ce_masked{}/featureNet_'.format(args.model_folder) + args.epoch_num + '.pth'))

# featureNet.cuda()
featureNet.eval()

# we dont need maskNet here right?
# maskNet = getattr(adversary, "MaskMan")(512)
# maskNet.load_state_dict(torch.load("saved_models/maskNet_19.pth"))
# maskNet.cuda()
# maskNet.eval()

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
landmark = {}
with open('data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

names_list = []
for folder in os.listdir('lfw'):
    imagenum = len(os.listdir('lfw/' + folder))
    if imagenum >= 2:
        names_list2 = os.listdir('lfw/' + folder)
        for i in range(len(names_list2)):
            names_list2[i] = folder + "/" + names_list2[i]
        names_list += names_list2

# img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])

print(len(names_list))
features_dict = {}
features = []
for idx, file_name in enumerate(names_list):
    if idx % 500 == 0:
        print(idx)
    img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(file_name),np.uint8),1),landmark[file_name])
    img1 = img1.transpose(2, 0, 1).reshape((1,3,112,96))
    img1 = (img1-127.5)/128.0
    img1 = Variable(torch.from_numpy(img1).float(),volatile=True)#.cuda()
    output = featureNet(img1)
    # print(output)
    output = fcNet(output)
    # print(output)
    # print(output.shape)
    features_dict[file_name] = output.data[0]
    # print(output.data[0])
    features.append(output.data[0].numpy())
    # if idx == 60:
    #     break
    # break
print("calculating cosine")

A = np.array(features)
similarity = np.dot(A, A.T)


# squared magnitude of preference vectors (number of occurrences)
square_mag = np.diag(similarity)

# inverse squared magnitude
inv_square_mag = 1 / square_mag

# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
inv_square_mag[np.isinf(inv_square_mag)] = 0

# inverse of the magnitude
inv_mag = np.sqrt(inv_square_mag)

# cosine similarity (elementwise multiply by inverse magnitudes)
cosine = similarity * inv_mag
cosine = cosine.T * inv_mag
# print(cosine)

top1accuracy = 0
k = 1
list_ = []
print("calculating top1 accuracy")
for i in range(len(cosine)):
    ind = np.argpartition(cosine[i], -(k+1))[-(k+1):]
    # print(ind)
    found = 0
    for idx in ind:
        if idx == i:
            continue
        # print(names_list[i].split("/"), names_list[idx].split("/"))
        if names_list[i].split("/")[0] == names_list[idx].split("/")[0]:
            found = 1
    list_.append(found)
print(np.mean(np.array(list_)))
import sys
sys.exit()
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

    img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
    img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])

    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    # output = net(img)
    output = featureNet(img)
    # print(output)
    output = fcNet(output)
    # print(output)
    f = output.data
    f1,f2 = f[0],f[2]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


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
