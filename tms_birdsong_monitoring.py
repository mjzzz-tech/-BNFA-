# 本文件用于执行天目山鸟鸣监测任务
import os
import time

import warnings

from cbam import CBAM_ResNet18
from noise_filtering_algorithm import noise_filtering
from utils import audio_augment

warnings.filterwarnings(action='ignore')
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageFile
import pandas as pd
import librosa
import numpy as np

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

# 挑选出最具有代表性的鸟类
most_represented_birds = ['Coccothraustes coccothraustes', 'Coloeus monedula', 'Cuculus canorus', 'Dendrocopos major',
                          'Garrulus glandarius', 'Hirundo rustica', 'Motacilla alba', 'Nycticorax nycticorax',
                          'Parus major', 'Passer montanus', 'Pica pica', 'Poecile palustris', 'Sitta europaea',
                          'Troglodytes troglodytes']

LABELS = most_represented_birds

model = CBAM_ResNet18().cpu()
# TODO - check exists
checkpoint = torch.load('/kaggle/input/test-model-b3/cbam_resnet18_tms.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
softmax_0 = nn.Softmax(dim=0)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dir = '/kaggle/input/birdtms-audio'


def automatic_recogintion(path, audio_name, loc):
    data = {'row_id': [], 'birds': []}
    file_name = audio_name.replace(".WAV", "")
    # Open file with Librosa
    # Split file into 5-second chunks
    # Extract spectrogram for each chunk
    # Predict on spectrogram
    # Get row_id and birds and store result
    # (maybe using a post-filter based on location)
    # The above steps are just placeholders, we will use mock predictions.
    # Our "model" will predict "nocall" for each spectrogram.
    sig, rate = librosa.load(path, sr=SAMPLE_RATE)
    # Split signal into 5-second chunks
    # Just like we did before (well, this could actually be a seperate function)
    sig_splits = []
    with tqdm(total=len(sig)) as pbar:
        for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
            pbar.update(int(SIGNAL_LENGTH * SAMPLE_RATE))
            split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

            # End of signal?
            if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
                break

            split = noise_filtering(split)
            split = audio_augment(split)
            sig_splits.append(split)
    # Get the spectrograms and run inference on each of them
    # This should be the exact same process as we used to
    # generate training samples!
    seconds, scnt = 0, 0
    with tqdm(total=len(sig_splits)) as pbar:
        for chunk in sig_splits:
            # Keep track of the end time of each chunk
            pbar.update(1)
            seconds += 5
            # Get the spectrogram
            hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
            mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                      sr=SAMPLE_RATE,
                                                      n_fft=2048,
                                                      hop_length=hop_length,
                                                      n_mels=SPEC_SHAPE[0],
                                                      fmin=FMIN,
                                                      fmax=FMAX)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # Normalize to match the value range we used during training.
            # That's something you should always double check!
            mel_spec -= mel_spec.min()
            mel_spec /= mel_spec.max()
            im = Image.fromarray(mel_spec * 255.0).convert("L")
            im = transform(im)
            # print(im.shape)
            im.unsqueeze_(0)
            # 没有这句话会报错
            im = im.to(device)
            # Predict
            p = model(im)[0]
            p = softmax_0(p)
            # print(p.shape)
            # Get highest scoring species
            idx = p.argmax()
            # print(idx)
            species = LABELS[idx]
            # print(species)
            score = p[idx]
            # print(score)
            # Prepare submission entry
            spath = path.split('/')[-1].rsplit('_', 1)[0]
            # print(spath)
            data['row_id'].append(path.split('/')[-1].rsplit('_', 1)[0] +
                                  '_' + str(seconds) + 's')
            # Decide if it's a "nocall" or a species by applying a threshold
            if score > 0.6:
                data['birds'].append(species)
                scnt += 1
            else:
                data['birds'].append('unknown')

    print('SOUNDSCAPE ANALYSIS DONE. FOUND {} BIRDS.'.format(scnt))
    # Make a new data frame and look at a few "results"
    results = pd.DataFrame(data, columns=['row_id', 'birds'])
    results.head()
    # Convert our results to csv
    file_output = 'result_' + loc + '_' + file_name + '.csv'
    results.to_csv(file_output, index=False)


for root, dirs, files in os.walk(input_dir):
    # with tqdm(total=len(files)) as pbar:
    for file in files:
        # pbar.update(1)
        since = time.time()
        audio_file_path = os.path.join(root, file)
        type_and_file = audio_file_path.replace("/kaggle/input/birdtms-audio/", "")
        birdtype = type_and_file.replace('/' + file, '')
        automatic_recogintion(audio_file_path, file, birdtype)

        time_elapsed = time.time() - since
        print('Preprocess complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
