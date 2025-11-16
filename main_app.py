# -*- coding: utf-8 -*-
# 加上这个声明，防止 Python 2 或某些环境无法识别中文注释
import tkinter as tk  # Python 标准 GUI 库
from tkinter import filedialog, messagebox, simpledialog  # 导入弹窗、文件选择框等组件
from tkinter import ttk  # 导入更美观的控件 (如 Tab页、进度条等)
import pygame  # 用于音频播放 (支持非阻塞播放，比系统自带播放器更好控制)
import os  # 用于操作系统路径处理
import librosa  # 专业音频处理库 (加载、重采样、去静音等)
import soundfile as sf  # 用于高质量保存 WAV 文件
import numpy as np  # 矩阵运算
import noisereduce as nr  # 专门的语音降噪库 (算法核心)

# --- 常用音频和系统库导入 ---
import pyaudio  # 用于控制麦克风进行实时录音
import wave  # 用于读取和写入 WAV 格式头信息
import joblib  # 用于保存和加载训练好的模型文件 (.pkl 或 .gmm)
from sklearn.mixture import GaussianMixture  # 高斯混合模型，说话人识别的核心算法
import threading  # 多线程库，防止录音时界面卡死
import time  # 用于延时 (sleep)

# --- V4.1 (Gemini 最终修正) ---
# 导入情感识别需要的 Transformer 深度学习库 (Hugging Face)
try:
    from transformers import (
        AutoConfig,  # 自动加载模型配置
        Wav2Vec2FeatureExtractor,  # 提取波形特征
        HubertPreTrainedModel,  # Hubert 预训练基类
        HubertModel  # Hubert 核心模型
    )
    import torch  # PyTorch 深度学习框架
    import torch.nn as nn  # 神经网络模块
    import torch.nn.functional as F  # 激活函数等

    TRANSFORMERS_AVAILABLE = True
    print("Hugging Face Transformers 库已找到。")
except ImportError:
    # 如果没安装这些库，打印错误但不让程序崩溃，只是禁用相关功能
    print(
        "错误：未找到 Hugging Face Transformers 库。请运行 'pip install transformers torch torchaudio accelerate'")
    TRANSFORMERS_AVAILABLE = False
except Exception as e:
    print(f"错误：Hugging Face 库导入失败: {e}")
    TRANSFORMERS_AVAILABLE = False
# --- 修改结束 ---

# --- 从本地文件导入 MFCC 特征提取函数 ---
try:
    from mfcc_coeff import extract_features  # 尝试导入刚才那个文件
except ImportError:
    print("错误：无法从 mfcc_coeff.py 导入 extract_features。请确保该文件存在且无误。")


    # 定义一个"替身"函数，万一导入失败，程序不会直接闪退，而是打印警告
    def extract_features(audio, sr):
        print("警告：extract_features 未成功导入，说话人识别功能将无法正常工作！")
        # 返回一个全零的矩阵，防止后续代码因为数据格式错误而报错
        return np.zeros((10, 13))

    # --- Baidu API 导入 (语音转文字) ---
try:
    from aip import AipSpeech  # 百度 AI 的 Python SDK
except ImportError:
    print("错误：未找到 baidu-aip 库。请运行 'pip install baidu-aip'")
    AipSpeech = None
import json

# --- Baidu API 凭证 (这里是你申请的账号密码) ---
APP_ID = '120502514'
API_KEY = 'KOkKtCZvc9g6GerlSbrXYSdZ'
SECRET_KEY = 'pUNnZ5XpDWh9IWDDiXKjBmePrlYsSHXS'

# 初始化 Baidu AipSpeech 客户端
if AipSpeech and APP_ID != '请替换为你自己的':
    try:
        client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)  # 建立连接对象
    except Exception as e:
        print(f"错误：初始化 Baidu AipSpeech 客户端失败: {e}")
        client = None
else:
    client = None
    if APP_ID == '请替换为你自己的':
        print("警告：Baidu API 凭证未填写，语音转文字功能将不可用。")

# --- 全局变量 (用于在不同函数之间传递数据) ---
selected_filepath = ""  # 记录当前用户在文件选择框选中的文件路径
enhanced_filepath = ""  # 记录经过降噪处理后生成的文件的路径
last_recorded_file = ""  # 记录最近一次麦克风录制的文件路径
stt_result_text = None  # 指向 GUI 上的文本框控件，用于显示语音转文字的结果

# --- STT (语音转文字) 录音控制变量 ---
stt_button = None  # 指向开始/停止按钮
stt_thread = None  # 记录正在运行的录音线程
stt_stop_event = threading.Event()  # 线程信号量，用于主线程通知子线程"该停了"

# --- GUI 控件变量 (为了能在函数里修改界面文字或状态) ---
status_label = None  # 底部状态栏文字
file_label = None  # 显示当前加载文件名的标签
playback_last_rec_button = None  # 回放按钮
play_enhanced_button = None  # 播放增强音频按钮
register_button = None  # 注册声纹按钮
recognize_button = None  # 识别声纹按钮
emotion_button = None  # 情感识别按钮
prop_decrease_slider = None  # 滑块：降噪比例
n_std_thresh_slider = None  # 滑块：噪声阈值
speaker_name_entry = None  # 输入框：注册人姓名

# --- 情感识别变量 ---
emotion_result_text = None  # 显示情感结果的文本框
emotion_model_choice = None  # (已弃用，固定使用中文模型)

# --- 模型配置 ---
SER_MODEL_ID = "xmj2002/hubert-base-ch-speech-emotion-recognition"  # 指定使用的预训练模型ID
# 缓存字典：把加载好的模型存在内存里，防止每次点击按钮都要重新加载模型(非常耗时)
loaded_models = {}
loaded_processors = {}


# ***************************************************************
# ****************** V4.1 (Gemini 最终修正) **********************
# * 核心修正：自定义模型类，用于加载 Hubert 模型
# * 为什么要这么写？因为 HuggingFace 的这个特定模型没有标准的配置文件
# * 所以我们需要手动用代码定义它的神经网络结构，才能正确加载权重。
# ***************************************************************

class HubertClassificationHead(nn.Module):
    """
    这是模型的'头部'，负责把 Hubert 提取的特征转换成 6 种情感的概率。
    """

    def __init__(self, config):
        super().__init__()
        # 全连接层：将特征维度映射到隐藏层维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)  # Dropout层防止过拟合
        # 输出层：将隐藏层映射到 num_labels (6个情感类别)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # 前向传播过程：线性层 -> 激活函数(Tanh) -> Dropout -> 输出层
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    """
    这是完整的模型类，包含 Hubert 主体和上面的分类头。
    """

    def __init__(self, config):
        super().__init__(config)
        # 强制设定类别数为 6 (怒, 惧, 喜, 中, 悲, 惊)
        config.num_labels = 6
        self.num_labels = config.num_labels

        # 加载 Hubert 主体模型 (提取声学特征)
        self.hubert = HubertModel(config)
        # 加载我们定义的分类头
        self.classifier = HubertClassificationHead(config)
        self.init_weights()  # 初始化权重

    def forward(self, x):
        # 1. 声音信号输入 Hubert
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        # 2. 池化 (Pooling)：因为输入语音长度不一，我们需要对时间维度求平均值，
        # 得到一个固定长度的向量来代表整句话
        x = torch.mean(hidden_states, dim=1)
        # 3. 输入分类头得到最终结果
        x = self.classifier(x)
        return x


# ***************************************************************
# ****************** V4.1 修正结束 ******************************
# ***************************************************************


# --- 录音函数 ---
def record_audio(output_filename, duration_seconds):
    """
    通用录音函数：录制指定时长的音频并保存为 WAV。
    参数：
        output_filename: 保存路径
        duration_seconds: 录音时长(秒)
    """
    CHUNK = 1024  # 缓冲区大小，每次读取 1024 个采样点
    FORMAT = pyaudio.paInt16  # 采样深度 16位整数 (标准CD音质深度)
    CHANNELS = 1  # 单声道 (语音识别通常只需要单声道)
    RATE = 16000  # 采样率 16kHz (Baidu API 和 Hubert 模型都要求这个采样率)

    p = pyaudio.PyAudio()  # 实例化 PyAudio 对象
    stream = None
    try:
        # 打开音频输入流
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,  # 设为 True 表示是录音(输入)
                        frames_per_buffer=CHUNK)

        frames = []
        if status_label:
            # 更新界面状态
            status_label.config(text=f"正在录音 ({duration_seconds}秒)...")
            root.update_idletasks()  # 强制立即刷新界面，防止界面看起来卡住

        # 循环读取音频数据
        # 循环次数 = 总采样点数 / 每次读取点数
        for i in range(0, int(RATE / CHUNK * duration_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)

        if status_label:
            status_label.config(text="录音结束，正在保存...")
            root.update_idletasks()

    except Exception as e:
        messagebox.showerror("录音错误", f"无法打开麦克风或录音失败: {e}")
        return None
    finally:
        # 无论成功与否，都要关闭流和释放资源，防止占用麦克风
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        try:
            p.terminate()
        except Exception:
            pass

    # 如果成功录到了数据，才保存文件
    if 'frames' in locals() and frames:
        try:
            # 确保文件夹存在，不存在则创建
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            # 打开 WAV 文件句柄
            wf = wave.open(output_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))  # 将二进制数据写入文件
            wf.close()
            return output_filename  # 返回文件名表示成功
        except Exception as e:
            messagebox.showerror("保存错误", f"保存录音文件失败: {e}")
            return None
    else:
        return None  # 录音失败


# --- 回放功能 ---
def play_audio_blocking(filename):
    """
    阻塞式播放音频。
    '阻塞'的意思是：在播放结束前，程序会停在这里，界面可能会暂时无法响应。
    用于短音频回放。
    """
    if not filename or not os.path.exists(filename):
        messagebox.showwarning("播放错误", "录音文件无效或不存在！")
        return

    p = None
    stream = None
    wf = None
    try:
        if status_label:
            status_label.config(text=f"正在回放: {os.path.basename(filename)}...")
            root.update_idletasks()

        CHUNK = 1024
        wf = wave.open(filename, 'rb')  # 以读模式打开 WAV
        p = pyaudio.PyAudio()
        # 打开输出流 (Output=True)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # 读取并播放，直到文件末尾
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)  # 写入声卡进行播放
            data = wf.readframes(CHUNK)

        # 清理资源
        try:
            stream.stop_stream(); stream.close()
        except:
            pass
        try:
            p.terminate()
        except:
            pass
        try:
            wf.close()
        except:
            pass

        if status_label: status_label.config(text="回放结束。")

    except Exception as e:
        if status_label: status_label.config(text=f"回放失败: {e}")
        # 确保出错也能关闭流
        try:
            stream.close()
        except:
            pass


def start_playback_thread():
    """
    启动回放线程。
    为了解决 'play_audio_blocking' 导致的界面卡顿，我们在一个新线程里调用它。
    """
    global last_recorded_file
    if not last_recorded_file or not os.path.exists(last_recorded_file):
        messagebox.showwarning("提示", "没有可回放的录音！")
        if playback_last_rec_button:
            playback_last_rec_button.config(state=tk.NORMAL)
        return

    # 播放开始前禁用按钮，防止用户重复点击
    if playback_last_rec_button:
        playback_last_rec_button.config(state=tk.DISABLED)

    # 启动线程，target 指向 play_and_reenable 函数
    threading.Thread(target=play_and_reenable, args=(last_recorded_file,), daemon=True).start()


def play_and_reenable(filename):
    """播放音频，播放完毕后重新启用按钮"""
    play_audio_blocking(filename)
    # 因为 GUI 只能在主线程更新，所以这里使用 root.after
    # 将更新按钮状态的任务加入主线程的消息队列
    if root and playback_last_rec_button:
        try:
            root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})
        except tk.TclError:
            pass  # 如果窗口已经关闭了，忽略错误


# --- 核心功能函数 ---
def load_file():
    """GUI 回调：点击'加载音频文件'按钮时触发"""
    global selected_filepath, enhanced_filepath, file_label, status_label, play_enhanced_button

    # 弹出系统文件选择框
    filepath = filedialog.askopenfilename(
        title="请选择一个 .wav 音频文件",
        filetypes=[("WAV files", "*.wav")]
    )
    if not filepath:
        return

    selected_filepath = filepath
    enhanced_filepath = ""  # 清空旧的增强文件路径
    filename = os.path.basename(filepath)
    if file_label:
        file_label.config(text=f"已加载: {filename}")
    if status_label:
        status_label.config(text="状态: 文件加载成功")
    # 新文件还没增强，所以禁用播放增强按钮
    if play_enhanced_button:
        play_enhanced_button.config(state=tk.DISABLED)


def play_sound(filepath):
    """
    使用 Pygame 播放音频。
    Pygame 适合播放长背景音乐或文件，它是非阻塞的（即点播放后代码继续往下走），
    所以不会卡住界面。
    """
    global status_label
    if not filepath:
        messagebox.showwarning("提示", "没有可播放的文件路径！")
        return
    if not os.path.exists(filepath):
        messagebox.showerror("错误", f"文件不存在: {filepath}")
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(filepath)  # 加载音乐
        pygame.mixer.music.play()  # 开始播放
        if status_label:
            if filepath == selected_filepath:
                status_label.config(text="状态: 正在播放 [原始] 音频...")
            elif filepath == enhanced_filepath:
                status_label.config(text="状态: 正在播放 [增强] 音频...")
            else:
                status_label.config(text=f"状态: 正在播放 {os.path.basename(filepath)}...")
            root.update_idletasks()
    except Exception as e:
        messagebox.showerror("播放错误", f"无法播放文件: {e}")


# --- 语音增强 (NR算法) ---
def enhance_voice_nr():
    """GUI 回调：点击'语音增强'按钮时触发"""
    global selected_filepath, enhanced_filepath, status_label, play_enhanced_button

    if not selected_filepath:
        messagebox.showwarning("提示", "请先加载一个音频文件！")
        return

    try:
        if status_label:
            status_label.config(text="状态: 正在进行 NR 降噪...请稍候...")
            root.update_idletasks()

        # 1. 加载音频，强制采样率为 16000 (sr=16000)
        y, sr = librosa.load(selected_filepath, sr=16000)

        # 2. 调用降噪核心逻辑
        enhanced_y = _process_nr(y, sr, noise_region='start')

        if enhanced_y is None:
            if status_label: status_label.config(text="状态: NR 降噪处理失败")
            return

        # 3. 构造保存路径 (在原文件名后加 _enhanced_nr)
        base_dir = os.path.dirname(selected_filepath)
        base_name = os.path.basename(selected_filepath)
        name, ext = os.path.splitext(base_name)
        enhanced_filename = f"{name}_enhanced_nr{ext}"
        enhanced_filepath = os.path.join(base_dir, enhanced_filename)

        # 4. 保存处理后的音频，指定为 PCM_16 格式 (通用性好)
        sf.write(enhanced_filepath, enhanced_y, sr, subtype='PCM_16')

        if status_label:
            status_label.config(text=f"状态: NR 语音增强完成！已保存至 {enhanced_filename}")
        # 处理完成，启用播放按钮
        if play_enhanced_button:
            play_enhanced_button.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("处理错误", f"NR 语音增强失败: {e}")
        enhanced_filepath = ""
        if play_enhanced_button: play_enhanced_button.config(state=tk.DISABLED)


def _process_nr(y, sr, noise_region='start'):
    """
    (V4.9 还原版) 降噪核心算法。
    使用 noisereduce 库的 Stationary (平稳) 噪声抑制算法。
    原理：假设音频的开头一段（1秒）全是噪音，计算这段噪音的频率特征（指纹），
    然后从整个音频中减去符合这个指纹的频率成分。
    """
    global status_label, prop_decrease_slider, n_std_thresh_slider
    try:
        # 1. 提取噪声样本 (Noise Clip)
        # 我们假设前 1.0 秒是纯背景噪声
        noise_duration = 1.0
        noise_samples = int(noise_duration * sr)

        if len(y) < noise_samples:
            # 如果文件太短，就用整个文件做噪声样本 (没办法的办法)
            noise_clip = y
        else:
            # 默认截取开头
            noise_clip = y[:noise_samples]

        # 2. 从界面滑块获取用户调节的参数
        prop_decrease = float(prop_decrease_slider.get())  # 降噪强度 (0~1)
        n_std_thresh_stat = float(n_std_thresh_slider.get())  # 阈值 (判断什么是噪音)

        # 3. 调用 noisereduce 库进行处理
        enhanced_y = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_clip,  # 传入必须的噪声样本
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh_stat  # 使用平稳模式参数
        )
        return enhanced_y
    except Exception as e:
        messagebox.showerror("降噪内部错误", f"_process_nr 失败: {e}")
        return None


# --- 说话人识别功能 ---
def start_register_thread():
    """GUI 回调：点击'注册声纹'按钮"""
    global speaker_name_entry
    speaker_name = speaker_name_entry.get()  # 获取输入的姓名
    if not speaker_name:
        messagebox.showwarning("提示", "请输入说话人姓名！")
        return
    # 禁用按钮防止重复点击
    register_button.config(state=tk.DISABLED)
    # 启动后台线程执行注册任务
    threading.Thread(target=register_speaker, args=(speaker_name,), daemon=True).start()


def register_speaker(speaker_name):
    """
    后台任务：注册说话人。
    流程：录音5次 -> 降噪 -> 提取特征 -> 拼接特征 -> 训练 GMM 模型 -> 保存模型
    """
    global status_label, last_recorded_file, playback_last_rec_button
    all_features = np.asarray(())  # 初始化空数组用于存放所有录音的特征
    try:
        temp_dir = "./temp_recordings"
        model_dir = "./gmm_models"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        record_count = 5  # 需要录制 5 段音频来覆盖不同的语调

        for i in range(record_count):
            if status_label:
                status_label.config(text=f"请为 {speaker_name} 录音 {i + 1}/{record_count} (4秒)...")
            time.sleep(0.5)  # 给用户半秒准备时间

            # 1. 录音
            raw_file = os.path.join(temp_dir, f"{speaker_name}_raw_{i + 1}.wav")
            recorded_path = record_audio(raw_file, 4)
            if recorded_path is None:
                # 如果某次录音失败，恢复按钮并退出
                if root and register_button:
                    root.after(0, register_button.config, {"state": tk.NORMAL})
                return

            last_recorded_file = recorded_path
            # 允许回放刚才录的声音
            if playback_last_rec_button:
                root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})

            # 2. 降噪预处理 (Pre-processing)
            # 加载刚录的音频
            y, sr = librosa.load(recorded_path, sr=16000)
            # 降噪 (能显著提高特征提取的纯净度)
            enhanced_y = _process_nr(y, sr, noise_region='start')

            if enhanced_y is None: continue

            # 3. 特征提取 (Feature Extraction)
            # 调用 mfcc_coeff.py 里的函数提取 MFCC
            vector = extract_features(enhanced_y, sr)

            if vector is None or vector.size == 0:
                print(f"警告：录音 {i + 1} 特征提取失败。")
                continue

                # 4. 特征拼接 (Stacking)
            # 将这 5 次录音提取到的特征矩阵在垂直方向堆叠起来
            # 形成一个巨大的特征矩阵，用于训练
            if all_features.size == 0:
                all_features = vector
            else:
                if vector.shape[1:] != all_features.shape[1:]:
                    continue  # 维度不对就跳过
                all_features = np.vstack((all_features, vector))

        # 检查是否提取到了有效特征
        if all_features.size == 0:
            if status_label:
                status_label.config(text="错误：未能从录音中提取任何有效特征，注册失败。")
            if root and register_button:
                root.after(0, register_button.config, {"state": tk.NORMAL})
            return

        if status_label:
            status_label.config(text=f"正在为 {speaker_name} 训练模型...")
            root.update_idletasks()

        # 5. 训练高斯混合模型 (GMM)
        # n_components=16: 使用 16 个高斯分布来拟合这个人的声纹特征分布
        # 这是一个经典的声纹识别配置
        gmm = GaussianMixture(n_components=16,
                              covariance_type='diag',  # 对角协方差矩阵，计算更快
                              max_iter=200,
                              random_state=0)
        gmm.fit(all_features)  # 开始拟合(训练)

        # 6. 保存模型
        model_path = os.path.join(model_dir, f"{speaker_name}.gmm")
        joblib.dump(gmm, model_path)  # 序列化保存到硬盘

        if status_label:
            status_label.config(text=f"成功！ {speaker_name} 的声纹已注册。")

    except Exception as e:
        messagebox.showerror("注册失败", f"发生错误: {e}")
    finally:
        if root and register_button:
            try:
                root.after(0, register_button.config, {"state": tk.NORMAL})
            except tk.TclError:
                pass


def start_recognize_thread():
    """GUI 回调：启动识别线程"""
    recognize_button.config(state=tk.DISABLED)
    threading.Thread(target=recognize_speaker, daemon=True).start()


def recognize_speaker():
    """
    后台任务：识别说话人。
    原理：录制一段音频，计算它在所有已注册模型下的得分(Likelihood)，得分最高者即为识别结果。
    """
    global status_label, last_recorded_file, playback_last_rec_button
    try:
        model_dir = "./gmm_models/"
        temp_dir = "./temp_recordings"
        os.makedirs(temp_dir, exist_ok=True)

        # 1. 加载所有模型
        gmm_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.gmm')]
        if not gmm_files:
            if status_label: status_label.config(text="错误: 声纹库为空！")
            if root and recognize_button: root.after(0, recognize_button.config, {"state": tk.NORMAL})
            return

        models = []
        speakers = []
        for f in gmm_files:
            try:
                models.append(joblib.load(f))  # 加载 GMM 对象
                speakers.append(os.path.basename(f).split('.gmm')[0])  # 提取文件名作为人名
            except Exception as e:
                print(f"警告：加载模型失败: {e}")

        # 2. 录制测试音频 (5秒)
        if status_label: status_label.config(text="请说话... 正在录制 5 秒音频...")
        time.sleep(0.5)
        test_file = os.path.join(temp_dir, "test_rec.wav")
        recorded_path = record_audio(test_file, 5)

        if not recorded_path:
            if root and recognize_button: root.after(0, recognize_button.config, {"state": tk.NORMAL})
            return

        last_recorded_file = recorded_path
        if playback_last_rec_button:
            root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})

        # 3. 预处理 (加载 -> 降噪)
        if status_label: status_label.config(text="正在处理和识别...")
        y, sr = librosa.load(recorded_path, sr=16000)
        enhanced_y = _process_nr(y, sr, noise_region='start')
        if enhanced_y is None: return

        # 4. 提取特征
        vector = extract_features(enhanced_y, sr)
        if vector is None or vector.size == 0:
            if status_label: status_label.config(text="识别失败：无法提取特征。")
            if root and recognize_button: root.after(0, recognize_button.config, {"state": tk.NORMAL})
            return

        # 5. 计算得分
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            # score_samples 计算每个特征帧属于该模型的概率(对数)
            scores = gmm.score_samples(vector)
            # sum() 将所有帧的得分加起来，作为整段音频的总得分
            log_likelihood[i] = scores.sum()

        # 6. 判决
        # 找到得分最大的那个索引 (argmax)
        winner_index = np.nanargmax(log_likelihood)
        winner_name = speakers[winner_index]
        max_likelihood = log_likelihood[winner_index]

        if status_label:
            status_label.config(text=f"识别成功！您应该是: {winner_name} (得分: {max_likelihood:.2f})")

    except Exception as e:
        messagebox.showerror("识别失败", f"发生错误: {e}")
    finally:
        if root and recognize_button:
            try:
                root.after(0, recognize_button.config, {"state": tk.NORMAL})
            except:
                pass


# --- 语音转文字 (Baidu API) ---
def recognize_speech_baidu(file_path):
    """
    核心函数：读取音频文件 -> 发送给百度云 -> 返回文字结果
    """
    global client, status_label
    if client is None:  # 检查客户端是否初始化成功
        messagebox.showerror("API 错误", "Baidu API 未成功初始化！")
        return None

    try:
        # 读取二进制音频数据
        with open(file_path, 'rb') as f:
            audio_data = f.read()

        # 调用 client.asr 进行识别
        # 'wav': 文件格式
        # 16000: 采样率
        # dev_pid=1537: 这是百度的普通话输入法模型 ID，最准
        result = client.asr(audio_data, 'wav', 16000, {'dev_pid': 1537, })

        # err_no 为 0 表示成功
        if result and result.get('err_no') == 0:
            return " ".join(result.get('result', []))  # 返回识别出的文本列表并拼接
        else:
            error_msg = result.get('err_msg', '未知错误')
            messagebox.showerror("API 识别失败", f"错误信息: {error_msg}")
            return None
    except Exception as e:
        messagebox.showerror("API 调用失败", f"异常: {e}")
        return None


def start_speech_to_text_thread():
    """
    GUI 回调：语音转文字按钮。
    功能：是一个开关，点一下开始录音，再点一下停止并上传。
    """
    global stt_thread, stt_button, stt_stop_event, status_label, stt_result_text

    # 如果线程正在运行，说明正在录音，现在的点击意味着"停止"
    if stt_thread and stt_thread.is_alive():
        if status_label: status_label.config(text="录音停止... 正在处理...")
        stt_stop_event.set()  # 发送停止信号
        if stt_button: stt_button.config(text="正在处理...", state=tk.DISABLED)
    else:
        # 否则，现在的点击意味着"开始"
        if client is None:  # 先检查 API 能不能用
            recognize_speech_baidu(None)
            return

        stt_stop_event.clear()  # 重置信号
        # 清空文本框
        if stt_result_text:
            stt_result_text.config(state=tk.NORMAL)
            stt_result_text.delete('1.0', tk.END)
            stt_result_text.config(state=tk.DISABLED)

        # 启动录音线程
        stt_thread = threading.Thread(target=recognize_speech_task, daemon=True)
        stt_thread.start()
        if stt_button:
            stt_button.config(text="正在录音... (点击停止)", state=tk.NORMAL, fg="red")


def recognize_speech_task():
    """
    后台任务：STT 完整流程。
    流式录音(直到停止) -> 保存 -> 降噪 -> 上传百度 -> 显示结果
    """
    global stt_button, last_recorded_file, stt_result_text, stt_stop_event, status_label, playback_last_rec_button
    p = None
    stream = None
    frames = []
    temp_dir = "./temp_recordings"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        if status_label:
            status_label.config(text="请说话... (点击按钮停止录音)")
            root.update_idletasks()

        # 循环读取录音，直到主线程设置了 stt_stop_event
        while not stt_stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError:
                pass

        # 停止录音流
        try:
            stream.stop_stream(); stream.close(); p.terminate()
        except:
            pass

        if len(frames) == 0: return

        # 1. 保存原始录音
        test_file = os.path.join(temp_dir, "stt_test_rec.wav")
        wf = wave.open(test_file, 'wb')
        wf.setnchannels(CHANNELS);
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT));
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        last_recorded_file = test_file
        if playback_last_rec_button: root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})

        if status_label: status_label.config(text="正在进行降噪处理...")

        # 2. 降噪处理 (提高识别率的关键步骤)
        y, sr = librosa.load(test_file, sr=16000)
        enhanced_y = _process_nr(y, sr, noise_region='start')

        audio_to_recognize = test_file  # 默认用原始的
        if enhanced_y is not None:
            enhanced_file = os.path.join(temp_dir, "stt_test_enhanced.wav")
            try:
                sf.write(enhanced_file, enhanced_y, sr, subtype='PCM_16')
                audio_to_recognize = enhanced_file  # 成功则用降噪后的
            except:
                pass

        if status_label: status_label.config(text="正在调用 Baidu API 进行语音识别...")

        # 3. 调用 API
        text_result = recognize_speech_baidu(audio_to_recognize)

        # 4. 显示结果 (跨线程更新 GUI)
        if text_result:
            if status_label: status_label.config(text="状态: 语音转文字成功！")

            def update_stt_text():
                stt_result_text.config(state=tk.NORMAL)
                stt_result_text.delete('1.0', tk.END)
                stt_result_text.insert(tk.END, text_result)
                stt_result_text.config(state=tk.DISABLED)

            try:
                root.after(0, update_stt_text)
            except:
                pass
        else:
            if status_label: status_label.config(text="状态: 语音转文字失败。")

    except Exception as e:
        if status_label: status_label.config(text="状态: 语音转文字异常终止")
    finally:
        # 恢复按钮状态
        stt_thread = None
        if root and stt_button:
            try:
                root.after(0, stt_button.config,
                           {"text": "8. 语音转文字 (点击开始)", "state": tk.NORMAL, "fg": "purple"})
            except:
                pass


# --- 语音情感识别 (Hugging Face / HuBERT) ---

def get_emotion_classifier():
    """
    加载情感识别模型。
    使用 Lazy Loading (懒加载) 策略：
    第一次调用时加载模型，之后直接从内存(loaded_models)读取，提高速度。
    """
    global loaded_models, loaded_processors, status_label
    model_id = SER_MODEL_ID

    # 1. 检查缓存
    if model_id in loaded_models and model_id in loaded_processors:
        return loaded_models[model_id], loaded_processors[model_id]

    if status_label:
        status_label.config(text=f"首次加载情感识别模型 ({model_id}), 请稍候...")
        root.update_idletasks()

    try:
        if not TRANSFORMERS_AVAILABLE:
            messagebox.showerror("库错误", "未找到 Transformers 库。")
            return None, None

        # 2. 加载处理器 (Feature Extractor)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        # 3. 加载模型配置和权重 (使用我们上面定义的自定义类)
        config = AutoConfig.from_pretrained(model_id)
        model = HubertForSpeechClassification.from_pretrained(model_id, config=config)

        # 4. 尝试把模型移到 GPU (如果有 N卡) 加速
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # 设为评估模式 (不进行梯度更新)

        # 5. 存入缓存
        loaded_models[model_id] = (model, device)
        loaded_processors[model_id] = processor

        if status_label: status_label.config(text=f"SER 模型加载成功!")
        return loaded_models[model_id], loaded_processors[model_id]

    except Exception as e:
        messagebox.showerror("模型加载失败", str(e))
        return None, None


def start_emotion_recognition_thread():
    """GUI 回调：启动情感识别"""
    if not TRANSFORMERS_AVAILABLE:
        messagebox.showerror("库错误", "库未安装")
        return
    emotion_button.config(state=tk.DISABLED)
    # 启动线程
    threading.Thread(target=recognize_emotion_task, daemon=True).start()


def recognize_emotion_task():
    """
    后台任务：情感识别完整流程。
    加载模型 -> 录音 -> 降噪 -> 深度学习预处理 -> 推理 -> 解析结果
    """
    global emotion_result_text, last_recorded_file, status_label, playback_last_rec_button

    # 1. 获取模型
    model_data, processor = get_emotion_classifier()
    if model_data is None:
        if root and emotion_button: root.after(0, emotion_button.config, {"state": tk.NORMAL})
        return

    model, device = model_data
    temp_dir = "./temp_recordings"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 2. 录音 4 秒
        rec_duration = 4
        test_file_path = os.path.join(temp_dir, "emotion_test_rec.wav")
        recorded_path = record_audio(test_file_path, rec_duration)
        if recorded_path is None:
            if root and emotion_button: root.after(0, emotion_button.config, {"state": tk.NORMAL})
            return

        last_recorded_file = recorded_path
        if playback_last_rec_button: root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})

        # 3. 降噪 (可选，但推荐)
        audio_input_path = recorded_path
        if status_label: status_label.config(text="正在降噪...")

        y, sr = librosa.load(recorded_path, sr=16000)
        enhanced_y = _process_nr(y, sr, noise_region='start')

        if enhanced_y is not None:
            enhanced_file_path = os.path.join(temp_dir, "emotion_test_enhanced.wav")
            sf.write(enhanced_file_path, enhanced_y, sr, subtype='PCM_16')
            audio_input_path = enhanced_file_path  # 使用降噪文件

        if status_label: status_label.config(text=f"正在推理...")

        # --- 关键：深度学习推理 ---
        try:
            # 加载音频到内存
            audio_array, sampling_rate = librosa.load(audio_input_path, sr=16000)

            # [优化 A] 去除首尾静音 (Trim)
            # top_db=25: 移除低于 25分贝 的部分
            audio_array, _ = librosa.effects.trim(audio_array, top_db=25)

            # [优化 B] 长度检查
            if len(audio_array) < 0.5 * 16000:  # 小于 0.5 秒可能导致模型报错
                if status_label: status_label.config(text="错误：有效语音太短")
                if root and emotion_button: root.after(0, emotion_button.config, {"state": tk.NORMAL})
                return

            # [优化 C] 音量归一化 (Normalize)
            # 将音量拉伸到标准范围，防止声音太小模型听不见
            audio_array = librosa.util.normalize(audio_array)

            # 预处理：将音频转换为模型能理解的 Tensor (张量)
            inputs_dict = processor(
                audio_array,
                sampling_rate=16000,
                padding=True,
                truncation=True,
                max_length=6 * 16000,  # 限制最大长度防止显存溢出
                return_tensors="pt"  # 返回 PyTorch 张量
            )

            # 移到 GPU
            input_values = inputs_dict.input_values.to(device)

            # 模型推理 (Forward Pass)
            with torch.no_grad():  # 此时不需要计算梯度，节省内存
                logits = model(input_values)

            # Softmax 归一化：将输出数值转换为概率 (0~1)
            probabilities = F.softmax(logits, dim=1)
            scores = probabilities[0].cpu().numpy()  # 转回 CPU 变成 numpy 数组

            # 标签映射字典
            id2label = {0: "anger", 1: "fear", 2: "happy", 3: "neutral", 4: "sad", 5: "surprise"}

            # 整理结果
            results_list = []
            for i in range(len(scores)):
                results_list.append({"label": id2label[i], "score": float(scores[i])})

            # 按置信度从高到低排序
            results_list.sort(key=lambda x: x["score"], reverse=True)

            # 显示结果
            best_emotion = results_list[0]
            display_text = f"识别结果: {best_emotion['label']} (置信度: {best_emotion['score']:.2f})\n\n详细结果:\n"
            for res in results_list:
                display_text += f"  - {res['label']}: {res['score']:.2f}\n"

            def update_emotion_text():
                emotion_result_text.config(state=tk.NORMAL)
                emotion_result_text.delete('1.0', tk.END)
                emotion_result_text.insert(tk.END, display_text)
                emotion_result_text.config(state=tk.DISABLED)

            if root: root.after(0, update_emotion_text)
            if status_label: status_label.config(text="状态: 语音情感识别成功！")

        except Exception as e:
            print(f"推理错误: {e}")
            if status_label: status_label.config(text="状态: 推理失败")

    except Exception as e:
        messagebox.showerror("错误", str(e))
    finally:
        if root and emotion_button:
            try:
                root.after(0, emotion_button.config, {"state": tk.NORMAL})
            except:
                pass


# --- GUI 界面创建 (主入口) ---
pygame.mixer.init()  # 初始化混音器
root = tk.Tk()  # 创建主窗口
root.title("数字音效处理系统 V4.9 (原始 noisereduce)")
root.geometry("550x750")  # 窗口大小

# 主容器
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = tk.Label(main_frame, text="数字音效处理系统", font=("Arial", 16, "bold"))
title_label.pack(pady=(0, 10))

# --- 状态显示区 ---
status_frame = tk.Frame(main_frame)
status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

file_label = tk.Label(status_frame, text="已加载: (无)", font=("Arial", 10), wraplength=500, anchor='w')
file_label.pack(fill=tk.X)

status_label = tk.Label(status_frame, text="状态: 准备就绪", font=("Arial", 10, "italic"), fg="blue", anchor='w')
status_label.pack(fill=tk.X, pady=(5, 0))

# --- Tab 选项卡控件 ---
tab_control = ttk.Notebook(main_frame)
tab1 = ttk.Frame(tab_control, padding=10)  # 页面1：文件处理
tab2 = ttk.Frame(tab_control, padding=10)  # 页面2：实时处理
tab_control.add(tab1, text='文件处理 (任务1)')
tab_control.add(tab2, text='实时处理 (任务2, 3, 4)')
tab_control.pack(expand=True, fill='both')

# --- 任务1: 语音增强 GUI ---
file_frame = tk.Frame(tab1, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
file_frame.pack(fill=tk.X, pady=5)
task1_label = tk.Label(file_frame, text="任务1: 语音增强 (处理文件)", font=("Arial", 12, "bold"))
task1_label.pack(anchor="w")


# [终极修正版] Task 1 现场录音逻辑
def record_for_task1():
    """
    Task 1 的现场录音按钮。
    特殊处理：在录音前必须先强制停止播放并释放文件锁，
    否则如果在播放 live_record.wav 时点击录音，会报错 "文件被占用"。
    """
    global selected_filepath, file_label, status_label, play_enhanced_button

    temp_dir = "./temp_recordings"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    filename = os.path.join(temp_dir, "live_record_enhancement.wav")

    # 关键：释放文件锁
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()  # Pygame 2.0+ 新特性
            except AttributeError:
                # 如果版本太旧没有 unload，只能暴力重启 mixer
                pygame.mixer.quit();
                pygame.mixer.init()
    except Exception as e:
        print(e)
    time.sleep(0.1)  # 等待系统释放

    # 录音
    if status_label: status_label.config(text="正在现场录音 (8秒)...")
    root.update_idletasks()
    filepath = record_audio(filename, 8)

    if filepath:
        selected_filepath = filepath
        global enhanced_filepath
        enhanced_filepath = ""
        if file_label: file_label.config(text=f"已现场采集: {os.path.basename(filename)}")
        if status_label: status_label.config(text="录音完成！请点击增强。")
        if play_enhanced_button: play_enhanced_button.config(state=tk.DISABLED)
    else:
        if status_label: status_label.config(text="录音失败")


# 按钮布局
record_task1_button = tk.Button(file_frame, text="0. 现场采集音频 (录音 8s)", command=record_for_task1,
                                font=("Arial", 12, "bold"), fg="red")
record_task1_button.pack(fill=tk.X, pady=(5, 0))

load_button = tk.Button(file_frame, text="1. 加载音频文件 (.wav)", command=load_file, font=("Arial", 12))
load_button.pack(fill=tk.X, pady=(5, 0))
play_original_button = tk.Button(file_frame, text="2. 播放原始音频", command=lambda: play_sound(selected_filepath),
                                 font=("Arial", 12))
play_original_button.pack(fill=tk.X, pady=5)

# 滑块区域
processing_frame = tk.Frame(file_frame)
processing_frame.pack(fill=tk.X, pady=5)
slider_frame_1 = tk.Frame(processing_frame)
slider_frame_1.pack(fill=tk.X, padx=5, pady=2)
tk.Label(slider_frame_1, text="降噪强度 (0-1):", font=("Arial", 10)).pack(side=tk.LEFT)
prop_decrease_slider = tk.Scale(slider_frame_1, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
prop_decrease_slider.set(1.0)
prop_decrease_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

slider_frame_2 = tk.Frame(processing_frame)
slider_frame_2.pack(fill=tk.X, padx=5, pady=2)
tk.Label(slider_frame_2, text="噪声阈值 (0-3):", font=("Arial", 10)).pack(side=tk.LEFT)
n_std_thresh_slider = tk.Scale(slider_frame_2, from_=0.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL)
n_std_thresh_slider.set(1.0)
n_std_thresh_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

enhance_button = tk.Button(processing_frame, text="3. 语音增强 (本地降噪)", command=enhance_voice_nr,
                           font=("Arial", 12, "bold"), fg="blue")
enhance_button.pack(fill=tk.X, pady=5)

play_enhanced_button = tk.Button(processing_frame, text="4. 播放增强后音频",
                                 command=lambda: play_sound(enhanced_filepath), font=("Arial", 12), state=tk.DISABLED)
play_enhanced_button.pack(fill=tk.X, pady=5)

# --- 任务2: 说话人识别 GUI ---
speaker_rec_frame = tk.Frame(tab2, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
speaker_rec_frame.pack(fill=tk.X, pady=10)
task2_label = tk.Label(speaker_rec_frame, text="任务2: 说话人识别 (GMM)", font=("Arial", 12, "bold"))
task2_label.pack(anchor="w")
register_frame = tk.Frame(speaker_rec_frame)
register_frame.pack(fill=tk.X, pady=5)
tk.Label(register_frame, text="说话人姓名:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(5, 5))
speaker_name_entry = tk.Entry(register_frame, font=("Arial", 10))
speaker_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
register_button = tk.Button(register_frame, text="5. 注册声纹 (录5次*4s)", command=start_register_thread,
                            font=("Arial", 12, "bold"), fg="green")
register_button.pack(side=tk.LEFT, padx=5)
recognize_button = tk.Button(speaker_rec_frame, text="6. 识别说话人 (录 5 秒)", command=start_recognize_thread,
                             font=("Arial", 12, "bold"), fg="red")
recognize_button.pack(fill=tk.X, pady=(10, 5))
playback_last_rec_button = tk.Button(speaker_rec_frame, text="7. 回放上次录音", command=start_playback_thread,
                                     font=("Arial", 12), state=tk.DISABLED)
playback_last_rec_button.pack(fill=tk.X, pady=5)

# --- 任务3 语音转文字 GUI ---
stt_frame = tk.Frame(tab2, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
stt_frame.pack(fill=tk.X, pady=10)
task3_label = tk.Label(stt_frame, text="任务3: 语音转文字 (Baidu API)", font=("Arial", 12, "bold"))
task3_label.pack(anchor="w")
stt_button = tk.Button(stt_frame, text="8. 语音转文字 (点击开始/停止)", command=start_speech_to_text_thread,
                       font=("Arial", 12, "bold"), fg="purple")
stt_button.pack(fill=tk.X, pady=5)
stt_result_text = tk.Text(stt_frame, height=3, font=("Arial", 10), state=tk.DISABLED, wrap=tk.WORD)
stt_result_text.pack(fill=tk.X, pady=5)

# --- 任务4 语音情感识别 GUI ---
emotion_rec_frame = tk.Frame(tab2, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
emotion_rec_frame.pack(fill=tk.X, pady=10)
task4_label = tk.Label(emotion_rec_frame, text="任务4: 语音情感识别 (HuBERT)", font=("Arial", 12, "bold"))
task4_label.pack(anchor="w")

model_choice_frame = ttk.LabelFrame(emotion_rec_frame, text="模型 (固定为 HuBERT)")
model_choice_frame.pack(fill=tk.X, padx=5, pady=5)
emotion_model_choice = tk.StringVar(value="Chinese")
rb_english = ttk.Radiobutton(model_choice_frame, text="英语 (禁用)", variable=emotion_model_choice, value="English",
                             state=tk.DISABLED)
rb_english.pack(anchor="w", padx=10)
rb_chinese = ttk.Radiobutton(model_choice_frame, text="中文 (默认)", variable=emotion_model_choice, value="Chinese",
                             state=tk.DISABLED)
rb_chinese.pack(anchor="w", padx=10)

emotion_button = tk.Button(emotion_rec_frame, text="9. 识别情感 (录 4 秒)", command=start_emotion_recognition_thread,
                           font=("Arial", 12, "bold"), fg="darkorange")
emotion_button.pack(fill=tk.X, pady=5)
emotion_result_text = tk.Text(emotion_rec_frame, height=5, font=("Arial", 10), state=tk.DISABLED, wrap=tk.WORD)
emotion_result_text.pack(fill=tk.X, pady=5)


# --- 退出处理 ---
def on_closing():
    print("正在关闭应用程序...")
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except Exception:
        pass
    try:
        root.destroy()
    except tk.TclError:
        pass


# --- 启动 ---
if __name__ == "__main__":
    root.protocol("WM_DELETE_WINDOW", on_closing)  # 绑定窗口关闭事件
    root.mainloop()  # 进入 GUI 主事件循环