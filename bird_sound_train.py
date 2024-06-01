# 本文件用于鸟声识别模型的训练
import os
import time
import random

import warnings

from utils import get_spectrograms

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from sklearn.utils import class_weight
from imblearn.under_sampling import RandomUnderSampler
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
import numpy as np
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
from torchinfo import summary

from cbam import CBAM_ResNet18
from noise_filtering_algorithm import noise_filtering

# 全局变量
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # seconds
SPEC_SHAPE = (224, 224)  # height x width
FMIN = 20
FMAX = 16000

# Load metadata file
train = pd.read_csv('/kaggle/input/birdclef-2023/train_metadata.csv')
train = train.query('rating>=4')

birds_count = {}
for bird_species, count in zip(train.primary_label.unique(),
                               train.groupby('primary_label')['primary_label'].count().values):
    birds_count[bird_species] = count

# 挑选出最具有代表性的鸟类
most_represented_birds = [key for key, value in birds_count.items() if value >= 197]





input_dir = '/kaggle/input/birdclef-2023/train_audio/'
output_dir = '/kaggle/working/melspectrogram_dataset/'
samples = []
# 使用 tqdm 包装 os.walk() 函数，显示进度条
since = time.time()
for root, dirs, files in os.walk(input_dir):
    for file in files:
        audio_file_path = os.path.join(root, file)
        bird_type = root.replace("/kaggle/input/birdclef-2023/train_audio/", "")
        if bird_type in most_represented_birds:
            samples += get_spectrograms(audio_file_path, bird_type, output_dir)

time_elapsed = time.time() - since
print('Preprocess complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# print(samples)
str_samples = ','.join(samples)
TRAIN_SPECS = shuffle(samples, random_state=RANDOM_SEED)
filename = open('a.txt', 'w')
filename.write(str_samples)
filename.close()
# print(samples)
str_samples = ','.join(samples)
TRAIN_SPECS = shuffle(samples, random_state=RANDOM_SEED)
filename = open('a.txt', 'w')
filename.write(str_samples)
filename.close()

from sklearn.model_selection import train_test_split
import shutil

filename = open('a.txt', 'r')
str_samples = filename.read()
filename.close()
str_samples = str_samples.replace("\\", "/")
samples = str_samples.split(',')
trainval_files, test_files = train_test_split(samples, test_size=0.1, random_state=42)

# 自定义训练集和验证集的路径，注意形式一致
train_dir = '/kaggle/working/train/'
val_dir = '/kaggle/working/val/'


def copyfiles(file, dir):
    filelist = file.split('/')
    filename = filelist[-1]
    lable = filelist[-2]
    cpfile = dir + "/" + lable
    if not os.path.exists(cpfile):
        os.makedirs(cpfile)
    cppath = cpfile + '/' + filename
    shutil.copy(file, cppath)


for file in trainval_files:
    copyfiles(file, train_dir)
for file in test_files:
    copyfiles(file, val_dir)

# 设置超参数
momentum = 0.9
BATCH_SIZE = 16
class_num = 18
EPOCHS = 50
lr = 1e-4
use_gpu = True
net_name = 'resnet18'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据转换
transform2 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform1 = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(size=224, scale=(0.67, 1)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 创建 ImageFolder 数据集
dataset_train = datasets.ImageFolder('/kaggle/working/train/', transform1)
dataset_val = datasets.ImageFolder('/kaggle/working/val/', transform2)

# 获取训练集和验证集的标签
train_labels = [label for _, label in dataset_train.samples]
val_labels = [label for _, label in dataset_val.samples]

# 创建 DataLoader 时指定采样器
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=ImbalancedDatasetSampler(dataset_train),
                          drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, sampler=ImbalancedDatasetSampler(dataset_val),
                        drop_last=True)

dset_sizes = len(dataset_train)
dset_sizes_val = len(dataset_val)
print(dset_sizes, dset_sizes_val)

model = CBAM_ResNet18()
'''
import torchvision.models as models

model = models.resnet18(pretrained=False)
# 修改最后线性层的输出通道数
model.fc = nn.Linear(512, len(most_represented_birds))
'''


def exp_lr_scheduler(optimizer, epoch, init_lr=4e-5, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    train_loss = []
    best_acc = 0.0
    # model_ft.train(True)
    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 进行模型训练
        model_ft.train()
        optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count = 0
        for data in train_loader:
            inputs, labels = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            if count % 240 == 0 or outputs.size()[0] < BATCH_SIZE:
                print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        print('Train_Loss: {:.4f} Train_Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # 进行模型验证
        model_ft.eval()
        with torch.no_grad():  # 无梯度，不进行调优
            val_loss = 0.0
            val_corrects = 0
            val_count = 0
            for data in val_loader:
                inputs, labels = data
                labels = torch.squeeze(labels.type(torch.LongTensor))
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.max(outputs, dim=1)[1]
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)
                val_count += 16

            val_loss = val_loss / dset_sizes_val
            val_acc = val_corrects.double() / dset_sizes_val
            print('Val_Loss: {:.4f} Val_Acc: {:.4f}'.format(
                val_loss, val_acc))

            # save model
        if epoch % 2 == 0:
            save_path = '/kaggle/working/model' + '_v' + str(epoch) + '.pth'
            torch.save(model_ft.state_dict(), save_path)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


criterion = nn.CrossEntropyLoss()
if use_gpu:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam((model.parameters()), lr=lr, weight_decay=1e-4)

# 模型的训练
train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)

# 进行模型验证
model.eval()
with torch.no_grad():  # 无梯度，不进行调优
    val_loss = 0.0
    val_corrects = 0
    val_count = 0
    for data in val_loader:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.max(outputs, dim=1)[1]
        val_loss += loss.item() * inputs.size(0)
        val_corrects += torch.sum(preds == labels)
        val_count += 16
        print(torch.sum(preds == labels))

    val_loss = val_loss / val_count
    val_acc = val_corrects.double() / val_count
    print('Val_Loss: {:.4f} Val_Acc: {:.4f}'.format(val_loss, val_acc))
