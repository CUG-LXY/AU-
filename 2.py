import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import csv
import matplotlib.pyplot as plt
import warnings
from torch.autograd import Variable
import re
warnings.filterwarnings("ignore")


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

class clf(nn.Module):
    def __init__(self, fc_hidden1=512, drop_p=0.3, num_classes=68):#num_classed要改
        super(clf, self).__init__()
        print('------ classification model-----')

        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p
        
        ## image
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=5,padding=2),              nn.ReLU(),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            nn.BatchNorm2d(64),
        )
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[1:-1]
        self.resnet = nn.Sequential(*modules)
        self.bn = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc = nn.Linear(fc_hidden1, num_classes)


        

    def forward(self, x):
        #print(x.shape)
        x = self.conv1_1(x)
        #x_2 = self.conv1_2(x)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        #x = self.resnet(x)
        # FC layers
        x = self.bn(F.relu(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc(x)

        return x

trans = transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize([256,256]),
        # transforms.Grayscale(1),
        # 转换成tensor向量
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class Dataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        "Initialization"

        images = []
        labels = []
        with  open(data_path, 'r') as f:
            for line in f.readlines():
                line.strip('\n')
                line.rstrip()
                information = line.split()
                images.append(information[0])
                labels.append([float(l) for l in information[1:len(information)]])

        self.data_path = data_path
        self.transform = transform
        self.images = images
        self.labels = labels

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images)


    def __getitem__(self, index):
        "Generates one sample of data"
        ImageName = self.images[index]
        label = self.labels[index]
        image = Image.open(ImageName)
        if transform is not None:
            image = transform(image)
        label = torch.FloatTensor(label)
        return image, label
        




if __name__ == '__main__':

    train_path=r'train_label.txt'
    train_data = Dataset(train_path,trans)
    train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True)



    net=clf()
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = nn.CrossEntropyLoss()
    loss_count = []
    for epoch in range(2):
        for i,(x,y) in enumerate(train_loader):
            # print(x,y)
            batch_x = Variable(x) 
            #print(batch_x.shape)# torch.Size([128, 1, 28, 28])
            # print(y)
            batch_y = Variable(y)#.flatten()
            #print(batch_y.shape)# torch.Size([128])
            
            out1 = net(batch_x) # torch.Size([128,10])
            # out2 = net(out1,batch_x)
            # 获取损失
            loss = loss_func(out1,batch_y)
            # 使用优化器优化损失
            optimizer.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            optimizer.step() # 将参数更新值施加到net的parmeters上

            if i % 10 == 0:
                print('Epoch:', epoch, '|Step:', i,
                  '|train loss:%.4f' % loss.data)


    test_path=r'test_label.txt'    
    test_data = Dataset(test_path,trans)
    test_loader = data.DataLoader(test_data,batch_size=12,shuffle=True)


    right = 0
    all = 0
    for i,(x,y) in enumerate(test_loader):

        batch_x = Variable(x) 
      
        batch_y = Variable(y)#.flatten()
    
        out = net(batch_x,batch_x) 
        pred_y = torch.max(out, 1)[1].data.squeeze()
        right += sum(pred_y==batch_y)
        all += batch_y.size(0)

    if all != 0:
        accuracy = right/all
        print('Test Acc: %.4f' % accuracy)
        
