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
import seaborn as sns
import math
def occlusion(featureNet, fcNet, image, image2, occ_size = 50, occ_stride = 50, occ_pixel = 0.5):
    #get the width and height of the image
    width, height = image.shape[-1], image.shape[-2]
    print(width, height)
    #setting the output image width and height
    output_height = int(math.ceil((height-occ_size)/occ_stride)) +1
    output_width = int(math.ceil((width-occ_size)/occ_stride)) +1
  
    #create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))
    # print(image.shape)
    #iterate all the pixels in each column
    print(output_height, output_width)
    for h in range(0, height):
        print(h)
        for w in range(0, width):
            print(w)
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            print(h_start, w_start, h_end, w_end)
            if (w_end) >= width or (h_end) >= height:
                continue
            # print(h, w)
            
            input_image = image.clone().detach()
            
            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            
            # output = input_image.data[:, :, :, :]
            # print(output.size(1))
            # fig, axarr = plt.subplots(output.size(1))
            # print(output.size())
            # for idx in range(output.size(1)):

            #     axarr[idx].imshow(output[0, idx])
            # plt.show()


            #run inference on modified image
            output = fcNet(featureNet(input_image))
            output2 = fcNet(featureNet(image2))
            
            # output = nn.functional.softmax(output, dim=1)
            # prob = output.tolist()[0][label]
            # prob = 
            #setting the heatmap location to probability value
            f1 = output.data[0]
            f2 = output2.data[0]
            cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
            prob = cosdistance


            heatmap[h, w] = prob 

    return heatmap

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
args = parser.parse_args()

predicts=[]
use_cuda = torch.cuda.is_available()



featureNet = getattr(net_sphere,args.net)()
featureNet.load_state_dict(torch.load('saved_models_ce/featureNet_' + args.epoch_num + '.pth'))
# featureNet.cuda()
featureNet.eval()

# we dont need maskNet here right?
maskNet = getattr(adversary, "MaskMan")()
maskNet.load_state_dict(torch.load("saved_models_ce/maskNet_" + args.epoch_num +".pth"))
# maskNet.cuda()
maskNet.eval()

fcNet = getattr(net_sphere, "fclayers")()
fcNet.load_state_dict(torch.load("saved_models_ce/fcNet_"+ args.epoch_num + ".pth"))
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
import matplotlib.pyplot as plt
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
    # print(img1)
    from matplotlib import pyplot as plt
    plt.imshow(img1)
    plt.show(block = False)
    print(img1.shape)
    # plt.imshow(img2)
    # import sys
    # sys.exit()
    
    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for i in range(len(imglist)):
        # print(imglist[i].transpose(2, 0, 1).shape)
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0

    img = np.vstack(imglist)
    # print(img[0].shape)
    # plt.imshow(img[0].transpose())
    
    if use_cuda:
        img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    else:
        img = Variable(torch.from_numpy(img).float(),volatile=True)

    # input_image = img[:1]
    # output = input_image.data[:, :, :, :]
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


    # heatmap = occlusion(featureNet, fcNet, img[:1], img[2:3], occ_size = 30, occ_stride = 5, occ_pixel = 0.5)
    # print(torch.max(heatmap).data)
    # imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=torch.max(heatmap).data)
    # figure = imgplot.get_figure()
    # print(heatmap)
    # print("done")
    output = featureNet(img)
    # print(output)
    print(output.shape)
    output = output.data[:4, 1, :, :]
    fig, axarr = plt.subplots(output.size(0))
    for idx in range(output.size(0)):
        # print(output[idx].unsqueeze(0).shape)
        print(output.shape)
        print(output[idx].shape)
        axarr[idx].imshow(output[idx])
    plt.show()
    image = img[0]
    # import sys
    # sys.exit()
    # output = fcNet(output)
    # f = output.data
    # f1,f2 = f[0],f[2]
    # cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    # predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


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
