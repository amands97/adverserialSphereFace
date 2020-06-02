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
    featureNet.load_state_dict(torch.load('saved_models_ce_mid/featureNet_' + args.epoch_num + '.pth'))
else:
    featureNet.load_state_dict(torch.load('saved_models_ce_mid{}/featureNet_'.format(args.model_folder) + args.epoch_num + '.pth'))

# featureNet.cuda()
featureNet.eval()

# we dont need maskNet here right?
# maskNet = getattr(adversary, "MaskMan")(512)
# maskNet.load_state_dict(torch.load("saved_models/maskNet_19.pth"))
# maskNet.cuda()
# maskNet.eval()

fcNet = getattr(net_sphere, "fclayers")()
if args.model_folder == -1:
    fcNet.load_state_dict(torch.load("saved_models_ce_mid/fcNet_"+ args.epoch_num + ".pth"))
else:
    fcNet.load_state_dict(torch.load("saved_models_ce_mid{}/fcNet_".format(args.model_folder)+ args.epoch_num + ".pth"))

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

train_names_list = []
test_names_list = []
for folder in os.listdir('data/ardata/'):
    if folder.startswith("."):
        continue
    if folder.endswith(".zip"):
        continue
    imagenum = len(os.listdir('data/ardata/' + folder))
    if imagenum >= 2:
        names_list2 = os.listdir('data/ardata/' + folder)
        for i in range(len(names_list2)):
            # check which session
            if int(names_list2[i].split(".")[0]) >= 14:
                continue
            elif names_list2[i] == "01.bmp":
                train_names_list.append(folder + "/"+ names_list2[i])
            else:
                test_names_list.append(folder + "/"+ names_list2[i])


# img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
# print(names_list)
print(len(train_names_list))
print(len(test_names_list))

# import sys
# sys.exit()
# features_dict = {}
train_features = []

for idx, file_name in enumerate(train_names_list):
    # if idx == 2:
    #     break
    if idx % 500 == 0:
        print(idx)
    img1 = cv2.imdecode(np.frombuffer(zfile.read(file_name),np.uint8),1)
    img1 = cv2.resize(img1, (96, 112))
    img1 = img1.transpose(2, 0, 1).reshape((1,3,112,96))
    img1 = (img1-127.5)/128.0
    img1 = Variable(torch.from_numpy(img1).float(),volatile=True)#.cuda()
    output = featureNet(img1)
    # print(output)
    output = fcNet(output)
    # print(output)
    # print(output.shape)
    # features_dict[file_name] = output.data[0]
    # print(output.data[0])
    train_features.append(output.data[0].numpy())
test_features = []
for idx, file_name in enumerate(test_names_list):
    # if idx ==18:
    #     break
    if idx % 500 == 0:
        print(idx)
    img1 = cv2.imdecode(np.frombuffer(zfile.read(file_name),np.uint8),1)
    img1 = cv2.resize(img1, (96, 112))

    img1 = img1.transpose(2, 0, 1).reshape((1,3,112,96))
    img1 = (img1-127.5)/128.0
    img1 = Variable(torch.from_numpy(img1).float(),volatile=True)#.cuda()
    output = featureNet(img1)
    # print(output)
    output = fcNet(output)
    # print(output)
    # print(output.shape)
    # features_dict[file_name] = output.data[0]
    # print(output.data[0])
    test_features.append(output.data[0].numpy())
print("calculating cosine")

# https://stackoverflow.com/questions/32688866/find-minimum-cosine-distance-between-two-matrices
B = np.array(train_features)
A = np.array(test_features)
dots = np.dot(A,B.T)
l2norms = np.sqrt(((A**2).sum(1)[:,None])*((B**2).sum(1)))
cosine_dists = 1 - (dots/l2norms)

# Get min values (if needed) and corresponding indices along the rows for res.
# Take care of zero L2 norm values, by using nanmin and nanargmin  
minval = np.nanmin(cosine_dists,axis=1)
cosine_dists[np.isnan(cosine_dists).all(1),0] = 0
res = np.nanargmin(cosine_dists,axis=1)
print(res)
total = 0
correct = 0
for i in range(len(res)):
    if res[i] == int(i / 6):
        correct += 1
    total += 1

print(correct* 100/(total * 1.0))        
# top1accuracy = 0
# k = 1
# list_ = []
# print("calculating top1 accuracy")
# name_start = ""
# for i in range(len(cosine)):
#     if names_list[i].split("/")[0] != name_start:
#         name_start = names_list[i].split("/")[0]
#     else:
#         continue
#     ind = np.argpartition(cosine[i], -(k+1))[-(k+1):]
#     # print(ind)
#     found = 0
#     for idx in ind:
#         if idx == i:
#             # print("asdas")
#             continue
#         # print(names_list[i].split("/"), names_list[idx].split("/"))
#         if names_list[i].split("/")[0] == names_list[idx].split("/")[0]:
#             found = 1
#     list_.append(found)
# print(np.mean(np.array(list_)))