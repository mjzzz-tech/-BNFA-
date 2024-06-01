# 本文件实现了噪声过滤算法
import torch
import numpy as np
import librosa
import pyAudioKits.audio as ak
import pyAudioKits.analyse as aly
import pyAudioKits.algorithm as alg
from train_noise_filtering_algorithm import stft
from train_noise_filtering_algorithm import net
from train_noise_filtering_algorithm import istft


def zero_pad_concat(inputs):
    '''
    zero_pad_concat 函数接受一个输入列表，返回一个填充后的 numpy 数组，其中所有输入数组都填充到相同的长度。
    '''
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t)
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat


def audio_pre_process(inputData):
    # inputData, sr = librosa.load(file, sr=32000, duration=5.01)
    inaudio = np.float32(inputData)
    mixed = torch.from_numpy(inaudio).type(torch.FloatTensor)
    mixeds = []
    mixeds.append(mixed)
    seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])
    x = torch.FloatTensor(zero_pad_concat(mixeds))
    batch = [x, seq_lens]
    return batch


def noise_filtering(inputData):
    test_mixed, seq_len = audio_pre_process(inputData)
    mixed = stft(test_mixed).unsqueeze(dim=1)
    real, imag = mixed[..., 0], mixed[..., 1]

    out_real, out_imag = net(real, imag)
    out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
    out_audio = istft(out_real, out_imag, test_mixed.size(1))
    out_audio = torch.squeeze(out_audio, dim=1)

    # 将输出音频的长度超过 seq_len 的部分置为零
    for i, l in enumerate(seq_len):
        out_audio[i, l:] = 0
    # 最终返回一个序列，可通过librosa库函数还原成音频
    return out_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()]


# 使用谱减法进行噪声过滤
def noise_filtering_specSubstract(filepath):
    data, rate = librosa.load(filepath, sr=32000, offset=None, duration=20.01)
    # soundfile.write("audio_file_denoised.ogg", data, samplerate=32000)
    f = ak.Audio(data, 32000)
    f1 = alg.specSubstract(f, f[f.getDuration() - 0.3:])
    f1.save(direction="audio_file_denoised.ogg")
