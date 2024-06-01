import json
import logging
import os
import shutil
import numpy as np
from tqdm import tqdm
import csv
import torch
from torch.utils import data
import librosa
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
import time
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from scipy.io import wavfile
from torch.utils.data import DataLoader
import soundfile as sf

'''
Params 类：这是一个从 JSON 文件加载超参数的类。它的构造函数接受一个 JSON 文件的路径，
读取该文件并将其内容存储为对象的属性。它还具有 save 方法，用于将当前对象的属性保存回 JSON 文件。
update 方法可以用于从另一个 JSON 文件中更新参数。dict 属性允许以字典形式访问 Params 实例的属性。
'''


class Params():

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


'''
RunningAverage 类：这是一个简单的类，用于计算某个数量的滑动平均值。
它的构造函数初始化步数和总和为 0，update 方法用于更新总和和步数，call 方法返回当前的平均值。
'''


class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


'''
set_logger 函数：这是一个设置日志记录的函数，它会将日志输出到终端和指定的日志文件。
它使用 Python 的 logging 模块实现，并设置了适当的格式。

'''


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


'''
save_dict_to_json 函数：这个函数用于将一个字典保存到 JSON 文件中。它的参数包括一个字典和一个 JSON 文件的路径。
在保存之前，它会将值的类型转换为浮点数，因为 JSON 不接受 numpy 数组等其他类型。
'''


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


'''
save_checkpoint 函数：这个函数用于保存模型和训练参数。它接收一个状态字典、一个布尔值 is_best 和一个保存路径。
它会将状态字典保存为 checkpoint + 'last.pth.tar'，如果 is_best=True，
则也会将其保存为 checkpoint + 'best.pth.tar'。函数还会检查保存路径是否存在，并在必要时创建该目录。
'''


def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


'''
load_checkpoint 函数：这个函数用于加载模型的参数和优化器的状态字典。它接收一个文件路径、模型和一个优化器（可选）。
它首先检查文件路径是否存在，然后加载状态字典并将其应用于模型。如果提供了优化器，它还会加载优化器的状态字典。
'''


def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder='/kaggle/input/dataset-bird-tms', setname='train'):
    '''
    load_data_list 函数：该函数用于加载数据集的文件列表，包括输入文件、输出文件和文件名。
    将数据集路径传入函数，函数会列出文件夹内所有 .wav 后缀的文件，然后将这些文件名称分别添加到输入、输出和文件名列表中。
    该函数返回一个字典对象，其中包含三个键：innames、outnames 和 shortnames。
    '''
    assert (setname in ['train', 'val'])

    dataset = {}
    foldername = folder + '/' + setname + 'set'

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    filelist = os.listdir("%s_noisy" % (foldername))
    filelist = [f for f in filelist if f.endswith(".ogg")]
    for i in tqdm(filelist):
        dataset['innames'].append("%s_noisy/%s" % (foldername, i))
        dataset['outnames'].append("%s_clean/%s" % (foldername, i))
        dataset['shortnames'].append("%s" % (i))

    return dataset


# DATA LOADING - LOAD FILE DATA
def load_data(dataset):
    '''
    load_data 函数：该函数用于加载数据集音频文件并转换为 numpy 数组。
    函数接受一个数据集字典作为输入，它首先初始化两个空的 numpy 数组（inaudio 和 outaudio）以存储输入和输出音频数据。
    然后使用 librosa 库的 load 方法加载音频文件，并将其转换为 np.float32 数据类型。
    最后，将这些数组添加到数据集字典中，并返回新的数据集字典。
    '''

    dataset['inaudio'] = [None] * len(dataset['innames'])
    dataset['outaudio'] = [None] * len(dataset['outnames'])

    for id in tqdm(range(len(dataset['innames']))):

        if dataset['inaudio'][id] is None:
            inputData, sr = librosa.load(dataset['innames'][id], sr=32000, duration=4.999)

            outputData, sr = librosa.load(dataset['outnames'][id], sr=32000, duration=4.999)

            shape = np.shape(inputData)

            dataset['inaudio'][id] = np.float32(inputData)
            dataset['outaudio'][id] = np.float32(outputData)

    return dataset


'''
AudioDataset 类：该类继承自 PyTorch 中的 Dataset 类，用于定义 PyTorch 模型训练过程中所需的数据集。
该类的构造函数接受一个字符串参数 data_type 来指定数据集的类型（train 或 val）。
它通过调用 load_data_list 和 load_data 函数，加载数据集列表和数据集音频数据。
然后定义了 getitem 方法，用于从数据集中获取某个样本。
该方法将输入和输出音频数据转换为 PyTorch 张量，并返回一个元组。
最后定义了 len 方法，返回数据集的长度。
'''


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):
        dataset = load_data_list(setname=data_type)
        self.dataset = load_data(dataset)

        self.file_names = dataset['innames']

    def __getitem__(self, idx):
        mixed = torch.from_numpy(self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        clean = torch.from_numpy(self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        '''
        zero_pad_concat 函数接受一个输入列表，返回一个填充后的 numpy 数组，其中所有输入数组都填充到相同的长度。
        '''
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0]] = inp
        return input_mat

    def collate(self, inputs):
        '''
        collate 函数接受一批输入，对每个输入进行 zero_pad_concat 处理，并将结果打包成一个列表返回，
        其中包括输入、输出和序列长度。
        '''
        mixeds, cleans = zip(*inputs)
        seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

        x = torch.FloatTensor(self.zero_pad_concat(mixeds))
        y = torch.FloatTensor(self.zero_pad_concat(cleans))

        batch = [x, y, seq_lens]
        return batch


'''
ISTFT 类，用于计算逆短时傅里叶变换（Inverse Short-Time Fourier Transform，ISTFT）。
ISTFT 是一种重构音频信号的方法，
它将短时傅里叶变换（Short-Time Fourier Transform，STFT）的结果转换为原始音频信号。
'''


class ISTFT(torch.nn.Module):
    def __init__(self, filter_length=2048, hop_length=574, window='hanning', center=True):
        '''
        在 ISTFT 类的构造函数中，首先初始化了一些参数，如滤波器长度、跳步长度、窗口类型等。
        然后使用 scipy.signal.get_window 函数获取指定类型和长度的窗口系数，
        并使用 inverse_stft_window 方法计算出窗口系数的逆值，备用于 ISTFT 运算中。
        最后，根据滤波器长度，生成傅里叶基函数 fourier_basis，
        对 fourier_basis 进行奇异值分解并取前一半的实部和虚部，得到一个矩阵用于后续计算。
        并使用 inverse_basis 存储上述矩阵与逆窗口系数的乘积，方便后续的运算。
        '''

        super(ISTFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.center = center

        win_cof = scipy.signal.get_window(window, filter_length)
        self.inv_win = self.inverse_stft_window(win_cof, hop_length)

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(self.inv_win * \
                                          np.linalg.pinv(fourier_basis).T[:, None, :])

        self.register_buffer('inverse_basis', inverse_basis.float())

    # Use equation 8 from Griffin, Lim.
    # Paper: "Signal Estimation from Modified Short-Time Fourier Transform"
    # Reference implementation: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/signal/spectral_ops.py
    # librosa use equation 6 from paper: https://github.com/librosa/librosa/blob/0dcd53f462db124ed3f54edf2334f28738d2ecc6/librosa/core/spectrum.py#L302-L311
    def inverse_stft_window(self, window, hop_length):
        '''
        inverse_stft_window 方法计算出窗口系数的逆值，备用于 ISTFT 运算中。
        '''
        window_length = len(window)
        denom = window ** 2
        overlaps = -(-window_length // hop_length)  # Ceiling division.
        denom = np.pad(denom, (0, overlaps * hop_length - window_length), 'constant')
        denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
        denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)
        return window / denom[:window_length]

    def forward(self, real_part, imag_part, length=None):
        '''
        在 forward 方法中，该方法接受两个张量 real_part 和 imag_part，分别表示 STFT 的实部和虚部。
        在 forward 方法中，首先将 real_part 和 imag_part 拼接成一个张量 recombined，
        然后通过卷积转置操作（F.conv_transpose1d）将 recombined 转换回原始音频信号。
        '''
        if (real_part.dim() == 2):
            real_part = real_part.unsqueeze(0)
            imag_part = imag_part.unsqueeze(0)

        recombined = torch.cat([real_part, imag_part], dim=1)

        inverse_transform = F.conv_transpose1d(recombined,
                                               self.inverse_basis,
                                               stride=self.hop_length,
                                               padding=0)

        padded = int(self.filter_length // 2)
        if length is None:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:-padded]
        else:
            if self.center:
                inverse_transform = inverse_transform[:, :, padded:]
            inverse_transform = inverse_transform[:, :, :length]

        return inverse_transform


# Utility functions for initialization
def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
    '''
    初始化函数，用于初始化复数权重矩阵。它使用了复数瑞利分布进行初始化。
    '''
    if not fanin:
        fanin = 1
        for p in W1.shape[1:]: fanin *= p
    scale = float(gain) / float(fanin)
    theta = torch.empty_like(Wr).uniform_(-math.pi / 2, +math.pi / 2)
    rho = np.random.rayleigh(scale, tuple(Wr.shape))
    rho = torch.tensor(rho).to(Wr)
    Wr.data.copy_(rho * theta.cos())
    Wi.data.copy_(rho * theta.sin())


# Layers
class ComplexConvWrapper(nn.Module):
    '''
    这是一个包装类，用于封装复数卷积层。它包含两个相同的卷积层，分别处理实部和虚部输入。
    '''

    def __init__(self, conv_module, *args, **kwargs):
        super(ComplexConvWrapper, self).__init__()
        self.conv_re = conv_module(*args, **kwargs)
        self.conv_im = conv_module(*args, **kwargs)

    def reset_parameters(self):
        fanin = self.conv_re.in_channels // self.conv_re.groups
        for s in self.conv_re.kernel_size: fanin *= s
        complex_rayleigh_init(self.conv_re.weight, self.conv_im.weight, fanin)
        if self.conv_re.bias is not None:
            self.conv_re.bias.data.zero_()
            self.conv_im.bias.data.zero_()

    def forward(self, xr, xi):
        real = self.conv_re(xr) - self.conv_im(xi)
        imag = self.conv_re(xi) + self.conv_im(xr)
        return real, imag


# Real-valued network module for complex input
class RealConvWrapper(nn.Module):
    '''
    这是一个包装类，用于封装实数卷积层。它只包含一个卷积层，处理实部和虚部的输入。
    '''

    def __init__(self, conv_module, *args, **kwargs):
        super(ComplexConvWrapper, self).__init__()
        self.conv_re = conv_module(*args, **kwargs)

    def forward(self, xr, xi):
        real = self.conv_re(xr)
        imag = self.conv_re(xi)
        return real, imag


class CLeakyReLU(nn.LeakyReLU):
    '''
    这是一个自定义的激活函数类，继承自nn.LeakyReLU。它对实部和虚部分别应用了F.leaky_relu函数。
    '''

    def forward(self, xr, xi):
        return F.leaky_relu(xr, self.negative_slope, self.inplace), \
               F.leaky_relu(xi, self.negative_slope, self.inplace)


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
class ComplexBatchNorm(torch.nn.Module):
    '''
    这是一个复数批归一化层的实现。它包含了复数参数的定义和初始化方法，以及前向传播方法。
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)
        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(num_features))
            self.register_buffer('RMi', torch.zeros(num_features))
            self.register_buffer('RVrr', torch.ones(num_features))
            self.register_buffer('RVri', torch.zeros(num_features))
            self.register_buffer('RVii', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


'''
基于复数神经网络的 Unet 模型，用于图像分割任务。整个模型包括编码器、解码器和获取比率掩模等操作。
'''


# NOTE: Use Complex Ops for DCUnet when implemented
# Reference:
#  > Progress: https://github.com/pytorch/pytorch/issues/755
def pad2d_as(x1, x2):
    '''
    对一个张量进行填充以匹配另一个张量的大小
    '''
    # Pad x1 to have same size with x2
    # inputs are NCHW
    diffH = x2.size()[2] - x1.size()[2]
    diffW = x2.size()[3] - x1.size()[3]

    return F.pad(x1, (0, diffW, 0, diffH))  # (L,R,T,B)


def padded_cat(x1, x2, dim):
    '''
    对两个张量进行跨度填充拼接
    '''
    # NOTE: Use torch.cat with pad instead when merged
    #  > https://github.com/pytorch/pytorch/pull/11494
    x1 = pad2d_as(x1, x2)
    x1 = torch.cat([x1, x2], dim=dim)
    return x1


class Encoder(nn.Module):
    '''
    编码器通过多次进行复数卷积（ComplexConvWrapper）、复数批量归一化（ComplexBatchNorm）和复数LeakyReLU激活函数，
    将输入图像不断降维，并保存每一层的特征图供解码器使用。
    '''

    def __init__(self, conv_cfg, leaky_slope):
        super(Encoder, self).__init__()
        self.conv = ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.bn = ComplexBatchNorm(conv_cfg[1])
        self.act = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        xr, xi = self.act(*self.bn(*self.conv(xr, xi)))
        return xr, xi


class Decoder(nn.Module):
    '''
    解码器则通过多次进行复数卷积转置（ComplexConvTranspose2d）、复数批量归一化和复数LeakyReLU激活函数，
    将编码器保存的特征图不断上采样，并与之前保存的特征图拼接在一起，最终输出分割结果。
    '''

    def __init__(self, dconv_cfg, leaky_slope):
        super(Decoder, self).__init__()
        self.dconv = ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.bn = ComplexBatchNorm(dconv_cfg[1])
        self.act = CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi, skip=None):
        if skip is not None:
            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.act(*self.bn(*self.dconv(xr, xi)))
        return xr, xi


class Unet(nn.Module):
    def __init__(self, cfg):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        for conv_cfg in cfg['encoders']:
            self.encoders.append(Encoder(conv_cfg, cfg['leaky_slope']))

        self.decoders = nn.ModuleList()
        for dconv_cfg in cfg['decoders'][:-1]:
            self.decoders.append(Decoder(dconv_cfg, cfg['leaky_slope']))

        # Last decoder doesn't use BN & LeakyReLU. Use bias.
        self.last_decoder = ComplexConvWrapper(nn.ConvTranspose2d,
                                               *cfg['decoders'][-1], bias=True)

        self.ratio_mask_type = cfg['ratio_mask']

    def get_ratio_mask(self, outr, outi):
        '''
        获取比率掩模的操作是为了进一步增强模型的表达能力，通过获取真实部分和虚部的比率掩模，
        将其乘以输入图像的幅度信息和相位信息，得到更精确的分割结果。
        '''

        def inner_fn(r, i):
            if self.ratio_mask_type == 'BDSS':
                return torch.sigmoid(outr) * r, torch.sigmoid(outi) * i
            else:
                # Polar cordinate masks
                # x1.4 slower
                mag_mask = torch.sqrt(outr ** 2 + outi ** 2)
                # M_phase = O/|O| for O = g(X)
                # Same phase rotate(theta), for phase mask O/|O| and O.
                phase_rotate = torch.atan2(outi, outr)

                if self.ratio_mask_type == 'BDT':
                    mag_mask = torch.tanh(mag_mask)
                # else then UBD(Unbounded)

                mag = mag_mask * torch.sqrt(r ** 2 + i ** 2)
                phase = phase_rotate + torch.atan2(i, r)

                # return real, imag
                return mag * torch.cos(phase), mag * torch.sin(phase)

        return inner_fn

    def forward(self, xr, xi):
        input_real, input_imag = xr, xi
        skips = list()

        for encoder in self.encoders:
            xr, xi = encoder(xr, xi)
            skips.append((xr, xi))

        skip = skips.pop()
        skip = None  # First decoder input x is same as skip, drop skip.
        for decoder in self.decoders:
            xr, xi = decoder(xr, xi, skip)
            skip = skips.pop()

        xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.last_decoder(xr, xi)

        xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
        ratio_mask_fn = self.get_ratio_mask(xr, xi)
        return ratio_mask_fn(input_real, input_imag)


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/kaggle/input/dataset-bird-tms/unet16.json',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=32, type=int, help='train epochs number')
args = parser.parse_args(args=[])

n_fft, hop_length = 2048, 574
window = torch.hann_window(n_fft).cuda()
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window, return_complex=False)
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1)  # Batch preserving sum for convenience.

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean ** 2) / (bsum(clean ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


json_path = os.path.join(args.model_dir)
params = Params(json_path)

net = Unet(params.model).cuda()
# TODO - check exists
checkpoint = torch.load('/kaggle/input/test-model-b3/DCUnet-tms-test.pth', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint)

train_dataset = AudioDataset(data_type='train')
# test_dataset = AudioDataset(data_type='val')
train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                               collate_fn=train_dataset.collate, shuffle=True, num_workers=0, drop_last=True)
# test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,collate_fn=test_dataset.collate, shuffle=False, num_workers=4)
torch.set_printoptions(precision=10, profile="full")

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=4e-5)
# Learning rate scheduler
scheduler = ExponentialLR(optimizer, 0.95)
for epoch in range(args.num_epochs):
    train_bar = tqdm(train_data_loader)

    cnt = 0
    since = time.time()
    for input in train_bar:
        cnt += 1
        train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
        mixed = stft(train_mixed).unsqueeze(dim=1)
        real, imag = mixed[..., 0], mixed[..., 1]

        out_real, out_imag = net(real, imag)
        out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
        out_audio = istft(out_real, out_imag, train_mixed.size(1))
        out_audio = torch.squeeze(out_audio, dim=1)
        for i, l in enumerate(seq_len):
            out_audio[i, l:] = 0

        '''
        sf.write('mixed.wav', train_mixed[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 32000)
        sf.write('clean.wav', train_clean[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 32000)
        sf.write('out.wav', out_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 32000)
        '''

        loss = wSDRLoss(train_mixed, train_clean, out_audio)
        if cnt % 90 == 0:
            print(epoch, loss)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + ' in this epoch')
    torch.save(net.state_dict(), '/kaggle/working/final.pth')
