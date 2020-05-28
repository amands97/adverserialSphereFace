# CUDA_VISIBLE_DEVICES=1 python lfw_eval.py --lfw lfw.zip --epoch_num 2
# it is already aligned!
# so remove the alignment code
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
fcNet = getattr(net_sphere, "fclayers")()


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


    if args.model_folder == -1:
        featureNet.load_state_dict(torch.load('saved_models_ce_masked/featureNet_' + args.epoch_num + '.pth'))
    else:
        featureNet.load_state_dict(torch.load('saved_models_ce_masked{}/featureNet_'.format(args.model_folder) + args.epoch_num + '.pth'))

    # we dont need maskNet here right?
    # maskNet = getattr(adversary, "MaskMan")(512)
    # maskNet.load_state_dict(torch.load("saved_models/maskNet_19.pth"))
    # maskNet.cuda()
    # maskNet.eval()

    if args.model_folder == -1:
        fcNet.load_state_dict(torch.load("saved_models_ce_masked/fcNet_"+ args.epoch_num + ".pth"))
    else:
        fcNet.load_state_dict(torch.load("saved_models_ce_masked{}/fcNet_".format(args.model_folder)+ args.epoch_num + ".pth"))

featureNet.cuda()
featureNet.eval()

fcNet.cuda()
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

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]
new_pairs_lines = []

for i in range(len(pairs_lines)):
    # if (i%100 == 0):
        # print("done:", i)
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
        try:
            zfile.read(name1)
            zfile.read(name2)

            new_pairs_lines.append(pairs_lines[i])

        except:
            # pairs_lines.pop(i)
            pass
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
        try:
            zfile.read(name1)
            zfile.read(name2)
            new_pairs_lines.append(pairs_lines[i])
        except:
            # pairs_lines.pop(i)
            pass
print(len(new_pairs_lines), len(pairs_lines))
pairs_lines = new_pairs_lines
for i in range(len(pairs_lines)):
    if (i%100 == 0):
        print("done:", i)
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        maskname1 = p[0]+'/np_mask_'+p[0]+'_'+'{:04}.jpg'.format(int(p[1])) + ".npy"
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
        maskname2 = p[0]+'/np_mask_'+p[0]+'_'+'{:04}.jpg'.format(int(p[2])) + ".npy"

    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        maskname1 = p[0]+'/np_mask_'+p[0]+'_'+'{:04}.jpg'.format(int(p[1])) + ".npy"

        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
        maskname2 = p[2]+'/np_mask_'+p[2]+'_'+'{:04}.jpg'.format(int(p[3])) + ".npy"

    # img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
    # img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])
    img1 = cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    img2 = cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    mask1 = np.frombuffer(zfile.read(maskname1), np.uint8)
    mask2 = np.frombuffer(zfile.read(maskname2), np.uint8)
    mask1 = cv2.resize(mask1, (96, 112))
    mask2 = cv2.resize(mask2, (96, 112))
    img1 = cv2.resize(img1, (96, 112))
    img2 = cv2.resize(img2, (96, 112))
    mask1 = mask1[..., np.newaxis]
    mask2 = mask2[..., np.newaxis]
    # print(img1.shape, mask1.shape)
    # imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    # maskList = [mask1, cv2.flip(mask1, 1), mask2, cv2.flip(mask2, 1)]
    imglist = [img1, img2]
    maskList = [mask1, mask2]

    for i in range(len(imglist)):
        # print(i)
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        maskList[i] = maskList[i].transpose(2, 0, 1).reshape((1, 1, 112, 96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    msk = np.vstack(maskList)
    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    # output = net(img)
    msk = Variable(torch.from_numpy(msk).float(),volatile=True).cuda()
    # mask = 
    # img = i

    # print(img.shape, msk.shape)
    # print((msk * img).shape)
    # img = msk * img
    img = img * msk
    for i in range(2):
        # print(mask[i, 0])
        # print(outimg[i])
        # image = cv2.resize(outimg[i].detach().unsqueeze(0).numpy(), (96, 112), interpolation = cv2.INTER_AREA)
        print(img[i].shape)
        # print(cv2.cvtColor(outimg[i].transpose(1, 2, 0), cv2.COLOR_BGR2RGB).shape)
        print("asdasd")
        image = img[i].transpose(1,2,0)
        # image = cv2.cvtColor(outimg[i].transpose(1,2,0), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (96, 112))
        image = image * 128.0 + 127.5
        # image = image.transpose((2, 0, 1))
        print(image.shape)
        # print(outimg[])
        cv2.imwrite("face" + str(i)+  ".jpg", image)

        # image.save("face" + str(i)+  ".jpg")

        # image = transforms.ToPILImage(mode='L')(mask[i])
        # image = image.resize((96, 112))
        # image.save("mask" + str(i)+  ".jpg")

    # import sys
    # sys.exit()
    output = featureNet(img)
    # print(output)
    output = fcNet(output)
    # print(output)
    f = output.data
    f1,f2 = f[0],f[1]
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))
    import sys
    sys.exit()

accuracy = []
thd = []
folds = KFold(n=len(pairs_lines), n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in enumerate(folds):
    
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
