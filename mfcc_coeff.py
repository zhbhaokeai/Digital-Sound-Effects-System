# -*- coding: utf-8 -*-
# 上面这行告诉电脑：这个文件里可能有中文，请用 UTF-8 编码来读取，防止乱码。

# --- 导入依赖库 ---
# import 是 Python 用来引入外部工具包的命令。
import librosa  # librosa 是一个非常强大的音频处理库，专门用来分析声音信号。
import numpy as np  # numpy 是 Python 进行科学计算的基础库，专门处理矩阵和数组运算。


# def 是 define 的缩写，用来定义一个函数（功能块）。
# 函数名是 extract_features，它接收两个参数：y 和 sr。
def extract_features(y, sr):
    """
    函数说明：
    从音频信号中提取特征，把声音变成一串数字，供机器识别。

    参数解释：
    y: 音频的时间序列数据（就是声音波形本身，是一个包含很多数字的列表）。
    sr: 采样率（Sample Rate），表示每秒钟采集多少个声音点，比如 16000。
    """

    # 1. 提取 20 维 MFCCs (梅尔频率倒谱系数)
    # MFCC 是模仿人耳听觉特性的特征，是语音识别中最常用的特征。
    # librosa.feature.mfcc 是库里自带的函数。
    # n_mfcc=20 表示我们要提取 20 个最重要的特征值。
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # 2. 提取 20 维 Delta-MFCCs (一阶差分)
    # 只有静态的 MFCC 是不够的，我们还想知道声音随时间变化的速率。
    # Delta 特征就是计算 MFCC 随时间的变化率（就像求导数一样）。
    # librosa.feature.delta 专门用来计算这个变化率。
    delta_mfccs = librosa.feature.delta(mfccs)

    # 3. 矩阵转置 (Transpose)
    # librosa 提取出来的矩阵形状是 (特征数量, 时间帧数)，比如 (20, 100)。
    # 但是后续的机器学习模型（sklearn）要求输入形状是 (样本数, 特征数量)，即 (100, 20)。
    # 所以我们需要用 .T 将矩阵“竖过来”（转置）。
    mfccs_t = mfccs.T
    delta_mfccs_t = delta_mfccs.T

    # 4. 特征拼接 (Horizontal Stack)
    # 现在我们有两个矩阵：MFCC 和 Delta。
    # 我们要把它们水平拼接到一起，让每一帧的时间点上既有 MFCC 特征，又有 Delta 特征。
    # np.hstack 是 numpy 库用来水平堆叠数组的函数。
    # 最终得到的 features 矩阵，每一行都有 40 个数字 (20 + 20)。
    features = np.hstack((mfccs_t, delta_mfccs_t))

    # return 表示函数执行完毕，把结果（特征矩阵）返回给调用它的地方。
    return features