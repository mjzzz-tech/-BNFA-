# 原文说明

该项目为文章《面向鸟鸣识别的背景噪声过滤算法设计与实现》的代码实现。算法使用Deep Complex U-net对含噪声的鸟鸣音频进行首次去噪，接着采用融合了CBAM注意力机制的ResNet18作为深度特征提取网络在特征提取过程中过滤噪声，可以有效提取鸟鸣深度特征，随后经过归一化的特征送入浅层分类器进行识别。本研究可以用于被动声学监测，通过被动声学监测技术，可以对自然界鸟类的种类、数量和分布进行长期、全面的监测，为生物多样性保护提供科学依据。

# 数据说明

本实验的相关数据由于数据过大无法上传至Github，如有需要可以通过下方链接下载：

天目山鸟鸣数据集
链接：https://pan.baidu.com/s/1bgum_KcL2IYURU7Bra7mDg 
提取码：brzi 
--来自百度网盘超级会员V5的分享

用于训练DCUnet的数据集
链接：https://www.kaggle.com/datasets/mjz00011/dataset-bird/

训练好的模型权重
https://www.kaggle.com/datasets/mjz00011/test-model-b3/
