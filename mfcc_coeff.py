# 文件名: mfcc_coeff.py
# -*- coding: utf-8 -*-
import librosa  # Python最强大的音频处理库
import numpy as np # 用于科学计算和矩阵操作

def extract_features(y, sr):
    """
    函数功能：从音频信号中提取特征向量，用于声纹识别。
    输入：
        y: 音频的时间序列数据 (一维数组)
        sr: 采样率 (每秒采样点数，例如 16000)
    输出：
        features: 组合后的特征矩阵 (帧数 x 40)
    """

    # 1. 提取 20 维 MFCCs (梅尔频率倒谱系数)
    # MFCC 是模拟人耳听觉特性的特征，能够很好地代表"音色"
    # n_mfcc=20: 提取最主要的20个系数，通常足够用于区分说话人
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # 2. 提取 20 维 Delta-MFCCs (一阶差分)
    # 单纯的 MFCC 只是静态特征，Delta 特征反映了语音随时间变化的速率
    # 这有助于捕捉说话人的语速、语调变化等动态特征
    delta_mfccs = librosa.feature.delta(mfccs)

    # 3. 矩阵转置 (Transpose)
    # librosa 提取出的形状是 (特征数 20, 帧数 N)
    # 机器学习库 sklearn 通常要求输入形状为 (样本数/帧数 N, 特征数 20)
    # 所以我们需要把矩阵"竖过来"
    mfccs_t = mfccs.T
    delta_mfccs_t = delta_mfccs.T

    # 4. 特征拼接 (Horizontal Stack)
    # 将 MFCC (N x 20) 和 Delta (N x 20) 在水平方向拼接
    # 最终得到一个 (N x 40) 的矩阵，每一帧都有 40 个特征值
    features = np.hstack((mfccs_t, delta_mfccs_t))

    return features