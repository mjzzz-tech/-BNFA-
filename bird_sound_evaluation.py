# 本文件用于将算法在数据集上进行测试
import os
import time
import warnings
import torch
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from cbam import CBAM_ResNet18
from utils import get_spectrograms_evaluation

warnings.filterwarnings(action='ignore')

import pandas as pd
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import prettytable
import itertools
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm

# Global vars
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

TRAIN = train.query('primary_label in @most_represented_birds')
LABELS = sorted(TRAIN.primary_label.unique())
TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)

print('FINAL NUMBER OF AUDIO FILES IN TRAINING DATA:', len(TRAIN))
# Parse audio files and extract training samples
input_dir = '/kaggle/input/birdclef2023-frog1/train_audio'
output_dir = '/kaggle/working/melspectrogram_dataset'
samples = []
with tqdm(total=len(TRAIN)) as pbar:
    for idx, row in TRAIN.iterrows():
        pbar.update(1)

        if row.primary_label in most_represented_birds:
            audio_file_path = os.path.join(input_dir, row.filename)
            get_spectrograms_evaluation(audio_file_path, row.primary_label, output_dir)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tp = {key: 0 for key in most_represented_birds}
fp = {key: 0 for key in most_represented_birds}
fn = {key: 0 for key in most_represented_birds}
y_pred = []  # ['2','2','3','1','4'] # 类似的格式
y_true = []  # ['0','1','2','3','4'] # 类似的格式


def test_model(model):
    sample_num = 0
    right_num = 0
    for root, dirs, files in os.walk('/kaggle/working/melspectrogram_dataset'):
        for file in files:
            sample_num = sample_num + 1
            file_path = os.path.join(root, file)

            try:
                im = Image.open(file_path)
            except Exception as e:
                print(file_path)
                continue

            type_and_file = file_path.replace("/kaggle/working/melspectrogram_dataset/", "")
            bird_type = type_and_file.replace('/' + file, '')
            im = transform(im)
            im.unsqueeze_(0)
            im = im.to(device)
            # Predict
            p = model(im)[0]
            # Get highest scoring species
            idx = p.argmax()
            species = LABELS[idx]
            y_pred.append(species)
            y_true.append(bird_type)

            if bird_type == species:
                right_num = right_num + 1
                tp[species] += 1
            else:
                fp[species] += 1
                fn[bird_type] += 1

            if sample_num % 100 == 0:
                acc = right_num / sample_num
                print('The accuracy is: ', acc, ' in ', sample_num, 'samples')

    accuracy = right_num / sample_num
    precision = {key: tp[key] / (tp[key] + fp[key]) for key in most_represented_birds}
    recall = {key: tp[key] / (tp[key] + fn[key]) for key in most_represented_birds}

    print('The accuracy is: ', accuracy)
    print('The precision for each class is:')
    print(precision)
    print('The recall for each class is:')
    print(recall)


model = CBAM_ResNet18().cpu()
# TODO - check exists
checkpoint = torch.load('/kaggle/input/test-model-b3/cabm_resnet18-v0322.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
'''
import torchvision.models as models

model = models.resnet18(pretrained=False).cpu()
# 修改最后线性层的输出通道数
model.fc = nn.Linear(512, len(most_represented_birds))
checkpoint = torch.load('/kaggle/input/test-model-b3/resnet18_v0320.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
'''
# 在评估模式下进行推断
with torch.no_grad():
    test_model(model)


def calculate_prediction_recall(label, pre, classes=None):
    """
    计算准确率和召回率:传入预测值及对应的真实标签计算
    :param label:标签
    :param pre:对应的预测值
    :param classes:类别名（None则为数字代替）
    :return:
    """

    # print(classes)
    confMatrix = confusion_matrix(label, pre)
    print(confMatrix)
    total_prediction = 0
    total_recall = 0
    result_table = prettytable.PrettyTable()
    class_multi = 1
    result_table.field_names = ['Type', 'Prediction(精确率)', 'Recall(召回率)', 'F1_Score']
    for i in range(len(confMatrix)):
        label_total_sum_col = confMatrix.sum(axis=0)[i]
        label_total_sum_row = confMatrix.sum(axis=1)[i]
        if label_total_sum_col:  # 防止除0
            prediction = confMatrix[i][i] / label_total_sum_col
        else:
            prediction = 0
        if label_total_sum_row:
            recall = confMatrix[i][i] / label_total_sum_row
        else:
            recall = 0
        if (prediction + recall) != 0:
            F1_score = prediction * recall * 2 / (prediction + recall)
        else:
            F1_score = 0
        result_table.add_row([classes[i], np.round(prediction, 3), np.round(recall, 3),
                              np.round(F1_score, 3)])

        total_prediction += prediction
        total_recall += recall
        class_multi *= prediction
    total_prediction = total_prediction / len(confMatrix)
    total_recall = total_recall / len(confMatrix)
    total_F1_score = total_prediction * total_recall * 2 / (total_prediction + total_recall)
    geometric_mean = pow(class_multi, 1 / len(confMatrix))
    # print(result_table)

    return total_prediction, total_recall, total_F1_score, result_table, geometric_mean, confMatrix


calculate_prediction_recall(y_true, y_pred, most_represented_birds)
C = confusion_matrix(y_true, y_pred, labels=most_represented_birds)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    混淆矩阵的可视化: 传入混淆矩阵和类别名（或数字代替）
    :param cm: 混淆矩阵
    :param classes: 类别
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confMatrix.jpg')
    plt.show()


plot_confusion_matrix(C, most_represented_birds)
