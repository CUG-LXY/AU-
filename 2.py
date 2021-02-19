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


# 是否使用gpu运算
use_gpu = torch.cuda.is_available()

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()




trans = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize(256),
        # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        transforms.RandomResizedCrop(227),
        # 图像用于翻转
        transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


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
        
train_path=r'train_label.txt'
train_data = Dataset(train_path,trans['train'])
train_loader = data.DataLoader(train_data,batch_size=10,shuffle=True)
test_path=r'test_label.txt'    
test_data = Dataset(test_path,trans['val'])
test_loader = data.DataLoader(test_data,batch_size=10,shuffle=True)
dataloaders = {'train':train_loader, 'val':test_loader}
dataset_sizes = {'train': train_data.__len__(), 'val': test_data.__len__()}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    Sigmoid_fun = nn.Sigmoid()
    since = time.time()
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 
        # 每训练一个epoch，验证一下网络模型
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0
 
            if phase == 'train':
                # 学习率更新方式
                scheduler.step()
                #  调用模型训练
                model.train()
 
                # 依次获取所有图像，参与模型训练或测试
                for data in dataloaders[phase]:
                    # 获取输入
                    inputs, labels = data
                    # 判断是否使用gpu
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
 
                    # 梯度清零
                    optimizer.zero_grad()
 
                    # 网络前向运行
                    outputs = model(inputs)
                    # 计算Loss值
                    loss = criterion(Sigmoid_fun(outputs), labels)
 
                    # 这里根据自己的需求选择模型预测结果准确率的函数
                    precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                    # precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)
                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    # 计算一个epoch的loss值和准确率
                    running_loss += loss.item() * inputs.size(0)
            else:
                # 取消验证阶段的梯度
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for data in dataloaders[phase]:
                        # 获取输入
                        inputs, labels = data
                        # 判断是否使用gpu
                        if use_gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
 
                        # 网络前向运行
                        outputs = model(inputs)
                        # 计算Loss值
                        # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                        loss = criterion(Sigmoid_fun(outputs), labels)
                        # 计算一个epoch的loss值和准确率
                        running_loss += loss.item() * inputs.size(0)
 
                        # 这里根据自己的需求选择模型预测结果准确率的函数
                        precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                        # precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)
                        running_precision += precision
                        running_recall += recall
                        batch_num += 1
 
            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            epoch_precision = running_precision / batch_num
            print('{} Precision: {:.4f} '.format(phase, epoch_precision))
            epoch_recall = running_recall / batch_num
            print('{} Recall: {:.4f} '.format(phase, epoch_recall))
            torch.save(model.state_dict(),'The_'+ str(epoch) + '_epoch_model.pkl'"Themodel_AlexNet.pkl")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
 
# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签
def calculate_acuracy_mode_one(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num
 
    return precision.item(), recall.item()
 
# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_acuracy_mode_two(model_pred, labels):
    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall
# 精调AlexNet
if __name__ ==  '__main__':
 
    # 导入Pytorch封装的AlexNet网络模型
    model = models.alexnet(pretrained=True)
    # 获取最后一个全连接层的输入通道数
    num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    feature_model = list(model.classifier.children())
    # 去掉原来的最后一层
    feature_model.pop()
    # 添加上适用于自己数据集的全连接层
    # 260数据集的类别数
    feature_model.append(nn.Linear(num_input, 260))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
    # 还可以为网络添加新的层
    # 重新生成网络的后半部分
    model.classifier = nn.Sequential(*feature_model)
    if use_gpu:
        model = model.cuda()
    # 定义损失函数
    criterion = nn.BCELoss()
 
    # 为不同层设定不同的学习率
    fc_params = list(map(id, model.classifier[6].parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    params = [{"params": base_params, "lr":0.0001},
              {"params": model.classifier[6].parameters(), "lr":0.001},]
    optimizer_ft = torch.optim.SGD(params, momentum=0.9)
 
    # 定义学习率的更新方式，每5个epoch修改一次学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
