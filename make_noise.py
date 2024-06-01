# 本代码用于向指定的音频数据集中添加噪声
from pydub import AudioSegment
from tqdm import tqdm
import os


# 创建目录
def create_directory(directory):
    os.makedirs(directory)


def add_noise(source_file_path, source_file):
    # 加载原始音频文件和背景音乐文件
    audio_file = AudioSegment.from_file(source_file_path)  # 原始音频文件
    # audio_file = audio_file + 9
    bg_music = AudioSegment.from_file('/kaggle/input/various-noise/frog.wav')  # 背景音乐文件

    bg_music = bg_music - 3  # 改变背景音乐文件的音量（需要灵活调整）
    bg_music = bg_music.set_frame_rate(audio_file.frame_rate)  # 设置背景音乐采样率为原始音频采样率

    # 将背景音乐混合到原始音频文件中
    mixed_audio = audio_file.overlay(bg_music, loop=True)

    # 导出混合后的音频文件
    # replace函数中的内容与遍历的目录路径相同，要加上“/”
    type_and_file = source_file_path.replace("/kaggle/input/birdclef-2023/train_audio/", "")

    # 指定的输出路径
    output = "/kaggle/working/birdclef-2023/train_audio/" + type_and_file
    directory = output.replace(source_file, "")
    if os.path.exists(directory):
        mixed_audio.export(output, format="ogg")
    else:
        create_directory(directory)
        mixed_audio.export(output, format="ogg")


# 遍历目录
def traverse_directory(directory):
    # 使用 tqdm 包装 os.walk() 函数，显示进度条
    for root, dirs, files in os.walk(directory):
        with tqdm(total=len(files)) as pbar:
            for file in files:
                pbar.update(1)
                file_path = os.path.join(root, file)
                # noise_filter(file_path, file)
                add_noise(file_path, file)
                # stop


# 调用函数创建目录
output_directory = "/kaggle/working/dataset_noisy/"
if os.path.exists(output_directory):
    # 调用函数遍历目录
    directory = "/kaggle/input/birdclef-2023/train_audio"  # 要遍历的目录路径
    traverse_directory(directory)
else:
    create_directory(output_directory)
    # 调用函数遍历目录
    directory = "/kaggle/input/birdclef-2023/train_audio"  # 要遍历的目录路径
    traverse_directory(directory)
