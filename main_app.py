# -*- coding: utf-8 -*-
# 这一行是声明文件编码为 UTF-8，确保代码中的中文注释能被正确识别，防止报错。

# --- 标准库和GUI库导入 ---
import tkinter as tk  # 导入 Python 自带的图形界面库，并给它起个别名叫 tk，方便后面少打字。
from tkinter import filedialog, messagebox, simpledialog  # 从 tkinter 里专门导入 文件选择框、弹窗提示、简单输入框 这三个组件。
from tkinter import ttk  # 导入 ttk 模块，这里面的控件（按钮、标签页）比 tk 自带的更好看。
import pygame  # 导入 pygame 库，这里主要用到它的混音器功能，用来播放音频文件（因为它不会卡住界面）。
import os  # 导入操作系统接口库，用于处理文件路径、创建文件夹等操作。
import threading  # 导入多线程库。这非常重要！因为录音很耗时，如果不放在新线程里，录音时界面就会卡死不动。
import time  # 导入时间库，用于让程序“暂停”一会儿（sleep）。
import json  # 导入 json 库，用于处理百度 API 返回的数据格式。

# --- 音频处理和数学运算库导入 ---
import librosa  # 导入专业的音频处理库，用于加载音频、重采样、去静音等复杂操作。
import soundfile as sf  # 导入 soundfile 库，它保存 WAV 文件的质量很高，且支持不同的编码格式。
import numpy as np  # 导入 numpy 库，它是 Python 数学运算的基础，主要用来处理音频数据的数组（矩阵）。
import noisereduce as nr  # 导入 noisereduce 库，这是一个专门用来给语音降噪的第三方工具。
import pyaudio  # 导入 PyAudio 库，它是专门用来控制麦克风进行实时录音的底层库。
import wave  # 导入 wave 库，用于读取和写入 WAV 文件的头部信息（比如采样率、声道数）。

# --- 机器学习库导入 ---
import joblib  # 导入 joblib 库，用于把训练好的模型保存到硬盘文件，或者从硬盘加载回内存。
from sklearn.mixture import GaussianMixture  # 从 sklearn 库导入 高斯混合模型 (GMM)，这是我们用来做说话人识别的核心算法。

# --- 深度学习库导入 (用于情感识别) ---
# try...except 是 Python 的错误捕获机制。
# 意思是：尝试运行 try 里面的代码，如果报错了，不要崩溃，而是运行 except 里面的代码。
try:
    from transformers import (
        AutoConfig,  # 用于自动加载模型的配置信息。
        Wav2Vec2FeatureExtractor,  # 用于从原始音频中提取特征，供 Transformer 模型使用。
        HubertPreTrainedModel,  # Hubert 模型的基类。
        HubertModel  # Hubert 模型的主体结构。
    )
    import torch  # 导入 PyTorch，这是目前最流行的深度学习框架之一。
    import torch.nn as nn  # 导入神经网络模块，用于构建自定义的模型层（比如全连接层）。
    import torch.nn.functional as F  # 导入神经网络的函数库，包含激活函数（如 softmax）等。

    TRANSFORMERS_AVAILABLE = True  # 设置一个标记，表示库加载成功。
    print("Hugging Face Transformers 库已找到。")  # 在控制台打印成功信息。
except ImportError:
    # 如果没有安装这些库，就会运行这里。
    print("错误：未找到 Hugging Face Transformers 库。请运行 'pip install transformers torch torchaudio accelerate'")
    TRANSFORMERS_AVAILABLE = False  # 标记库加载失败，后续代码会禁用相关功能。
except Exception as e:
    # 捕获其他未知的错误。
    print(f"错误：Hugging Face 库导入失败: {e}")
    TRANSFORMERS_AVAILABLE = False

# --- 导入自定义模块 ---
try:
    from mfcc_coeff import extract_features  # 尝试从我们要刚才写的 mfcc_coeff.py 文件中导入 extract_features 函数。
except ImportError:
    print("错误：无法从 mfcc_coeff.py 导入 extract_features。请确保该文件存在且无误。")


    # 如果导入失败，为了防止程序崩溃，我们在这里定义一个“替身”函数。
    # 这个函数什么都不做，只返回一个全零的矩阵，保证程序能跑下去。
    def extract_features(audio, sr):
        print("警告：extract_features 未成功导入，说话人识别功能将无法正常工作！")
        return np.zeros((10, 13))  # 返回一个 10行13列 的零矩阵作为占位符。

# --- Baidu API 导入 (语音转文字) ---
try:
    from aip import AipSpeech  # 尝试导入百度 AI 的 Python SDK。
except ImportError:
    print("错误：未找到 baidu-aip 库。请运行 'pip install baidu-aip'")
    AipSpeech = None  # 如果没找到库，把 AipSpeech 设为 None，后面会检查这个变量。

# --- Baidu API 凭证配置 ---
# 这些是连接百度云服务器的“钥匙”，需要去百度 AI 开放平台申请。
# 这里的 key 是您提供的，实际发布时应该删掉保护隐私。
APP_ID = '120502514'
API_KEY = 'KOkKtCZvc9g6GerlSbrXYSdZ'
SECRET_KEY = 'pUNnZ5XpDWh9IWDDiXKjBmePrlYsSHXS'

# 初始化 Baidu AipSpeech 客户端
# 只有当库导入成功 (AipSpeech 存在) 且 APP_ID 不是默认提示文字时，才初始化。
if AipSpeech and APP_ID != '请替换为你自己的':
    try:
        client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)  # 创建一个客户端对象，用于发送请求。
    except Exception as e:
        print(f"错误：初始化 Baidu AipSpeech 客户端失败: {e}")
        client = None
else:
    client = None
    if APP_ID == '请替换为你自己的':
        print("警告：Baidu API 凭证未填写，语音转文字功能将不可用。")

# --- 全局变量定义 ---
# 这些变量需要在不同的函数（按钮点击事件）之间共享数据，所以定义在最外面。
selected_filepath = ""  # 存储用户刚才在电脑里选中的那个 WAV 文件的路径。
enhanced_filepath = ""  # 存储降噪处理之后生成的新文件的路径。
last_recorded_file = ""  # 存储最近一次使用麦克风录制的文件路径。
stt_result_text = None  # 这个变量用来指向界面上的一个文本框，用于显示语音转文字的结果。

# --- STT (语音转文字) 录音控制变量 ---
stt_button = None  # 指向界面上的“语音转文字”按钮，方便我们改它的文字（比如变成“停止录音”）。
stt_thread = None  # 用来存储正在运行的录音线程。
stt_stop_event = threading.Event()  # 这是一个线程信号灯。主线程设置它为 True，子线程检测到就会停止录音。

# --- GUI 控件变量 ---
# 这些变量用来存储界面上的各种标签、按钮、滑块，方便我们在代码里修改它们的状态（比如禁用按钮、更新文字）。
status_label = None  # 底部显示“状态：正在录音...”的标签。
file_label = None  # 显示“已加载：xxx.wav”的标签。
playback_last_rec_button = None  # 回放按钮。
play_enhanced_button = None  # 播放增强后音频的按钮。
register_button = None  # 注册声纹按钮。
recognize_button = None  # 识别声纹按钮。
emotion_button = None  # 情感识别按钮。
prop_decrease_slider = None  # 滑块：控制降噪强度。
n_std_thresh_slider = None  # 滑块：控制噪声阈值。
speaker_name_entry = None  # 输入框：用户输入名字的地方。

# --- 情感识别变量 ---
emotion_result_text = None  # 指向用于显示情感识别结果（比如“愤怒：90%”）的文本框。
emotion_model_choice = None  # (这个变量暂时没用上，原本设计用来选语言)。

# --- 模型配置 ---
SER_MODEL_ID = "xmj2002/hubert-base-ch-speech-emotion-recognition"  # 这是我们在 HuggingFace 上选用的预训练模型名字。
# 这两个字典用来做“缓存”。
# 因为模型加载很慢，我们加载一次后就把模型存在这里，下次直接用，不用再加载了。
loaded_models = {}
loaded_processors = {}


# ***************************************************************
# ****************** 自定义模型类定义 ****************************
# * 这一部分是深度学习的核心。因为 HuggingFace 的这个特定模型没有
# * 标准的配置文件，我们需要手动写代码告诉电脑这个模型的神经网络长什么样。
# ***************************************************************

class HubertClassificationHead(nn.Module):
    """
    这个类定义了模型的“头部”（Head）。
    Hubert 模型提取出通用的声音特征后，需要这个头部把特征分类成“愤怒”、“高兴”等情感。
    """

    def __init__(self, config):
        super().__init__()  # 初始化父类。
        # 定义一个全连接层 (Linear)，把特征维度 (hidden_size) 映射到同样的维度。
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 Dropout 层，随机丢弃一些神经元，防止模型“死记硬背”（过拟合）。
        self.dropout = nn.Dropout(config.classifier_dropout)
        # 定义输出层，把特征映射到最终的类别数量 (num_labels，这里是6种情感)。
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # 这是数据流动的过程：
        x = self.dense(x)  # 1. 经过全连接层。
        x = torch.tanh(x)  # 2. 经过激活函数 Tanh，把数值压缩到 -1 到 1 之间。
        x = self.dropout(x)  # 3. 经过 Dropout。
        x = self.out_proj(x)  # 4. 经过输出层，得到 6 个情感的得分。
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    """
    这个类定义了完整的模型结构。
    它包含了 Hubert 主体（负责听声音）和上面的 ClassificationHead（负责判断情感）。
    """

    def __init__(self, config):
        super().__init__(config)  # 初始化配置。
        # 强制设定分类的数量为 6 (对应 Anger, Fear, Happy, Neutral, Sad, Surprise)。
        config.num_labels = 6
        self.num_labels = config.num_labels

        # 加载 Hubert 主体模型。
        self.hubert = HubertModel(config)
        # 加载我们上面定义的分类头。
        self.classifier = HubertClassificationHead(config)
        self.init_weights()  # 初始化神经网络的权重参数。

    def forward(self, x):
        # 这是整个模型的推理过程：
        # 1. 把声音数据 x 喂给 Hubert 模型。
        outputs = self.hubert(x)
        # 2. 取出 Hubert 的输出（Hidden States）。
        hidden_states = outputs[0]
        # 3. 池化 (Pooling)：因为说话时长不一样，特征长度也不一样。
        # 我们对时间维度求平均值 (mean)，把变长的特征变成固定长度的向量。
        x = torch.mean(hidden_states, dim=1)
        # 4. 把这个向量喂给分类头，得到最终的情感得分。
        x = self.classifier(x)
        return x


# ***************************************************************
# ****************** 核心功能函数实现 ****************************
# ***************************************************************

# --- 录音函数 ---
def record_audio(output_filename, duration_seconds):
    """
    功能：控制麦克风录音，并保存为 WAV 文件。
    参数：
        output_filename: 文件保存的路径。
        duration_seconds: 录音时长（秒）。
    """
    CHUNK = 1024  # 缓冲区大小。意思是每次从麦克风读取 1024 个数据点。
    FORMAT = pyaudio.paInt16  # 采样深度。16位整数是 CD 音质的标准深度。
    CHANNELS = 1  # 声道数。1 代表单声道，语音识别通常不需要立体声。
    RATE = 16000  # 采样率。16000Hz 是目前语音识别的主流标准。

    p = pyaudio.PyAudio()  # 创建一个 PyAudio 对象，用于操作音频硬件。
    stream = None
    try:
        # 打开音频流（相当于打开麦克风）。
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,  # input=True 表示我们要“输入”声音（录音）。
                        frames_per_buffer=CHUNK)

        frames = []  # 创建一个空列表，用来存放录下来的数据块。

        # 更新界面上的状态文字。
        if status_label:
            status_label.config(text=f"正在录音 ({duration_seconds}秒)...")
            root.update_idletasks()  # 强制刷新界面，否则界面可能会卡住不动。

        # 开始循环读取数据。
        # 循环次数 = (每秒采样点数 / 每次读取点数) * 秒数。
        for i in range(0, int(RATE / CHUNK * duration_seconds)):
            data = stream.read(CHUNK)  # 从麦克风读取一块数据。
            frames.append(data)  # 把数据加到列表里。

        # 录音结束，更新界面。
        if status_label:
            status_label.config(text="录音结束，正在保存...")
            root.update_idletasks()

    except Exception as e:
        # 如果中间出错了（比如没插麦克风），弹窗提示错误。
        messagebox.showerror("录音错误", f"无法打开麦克风或录音失败: {e}")
        return None  # 返回 None 表示录音失败。
    finally:
        # finally 里的代码无论成功还是报错都会执行。
        # 这里用来清理资源，关闭麦克风，防止程序占用设备。
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

    # 只有当成功录到了数据 (frames 不为空)，才保存文件。
    if 'frames' in locals() and frames:
        try:
            # 确保保存文件的文件夹存在，不存在就创建它。
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            # 打开一个 WAV 文件准备写入。
            wf = wave.open(output_filename, 'wb')
            wf.setnchannels(CHANNELS)  # 设置声道。
            wf.setsampwidth(p.get_sample_size(FORMAT))  # 设置采样深度。
            wf.setframerate(RATE)  # 设置采样率。
            wf.writeframes(b''.join(frames))  # 把列表里的数据拼起来写入文件。
            wf.close()  # 关闭文件。
            return output_filename  # 返回保存的文件路径，表示成功。
        except Exception as e:
            messagebox.showerror("保存错误", f"保存录音文件失败: {e}")
            return None
    else:
        return None  # 如果没有数据，返回失败。


# --- 回放功能 ---
def play_audio_blocking(filename):
    """
    功能：播放音频文件。
    注意：这是“阻塞式”播放，也就是说播放时程序会停在这里等待播放结束，界面可能会暂时没反应。
    适合播放很短的提示音或录音片段。
    """
    # 先检查文件是否存在。
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
        wf = wave.open(filename, 'rb')  # 以只读模式 ('rb') 打开 WAV 文件。
        p = pyaudio.PyAudio()

        # 打开音频输出流（相当于打开扬声器）。
        # 参数都是从 WAV 文件头里读取出来的，保证播放格式正确。
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)  # output=True 表示我们要“输出”声音。

        # 循环读取并播放。
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)  # 把数据写入声卡播放。
            data = wf.readframes(CHUNK)

        # 播放完毕，清理资源。
        try:
            stream.stop_stream();
            stream.close()
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
        # 如果播放出错，更新状态栏并在后台关闭流。
        if status_label: status_label.config(text=f"回放失败: {e}")
        try:
            stream.close()
        except:
            pass


def start_playback_thread():
    """
    功能：启动一个新的线程来播放录音。
    原因：因为 play_audio_blocking 会卡住界面，所以我们在一个新线程里调用它，
    这样主界面依然可以响应鼠标点击。
    """
    global last_recorded_file
    # 如果没有录音文件，弹窗提示。
    if not last_recorded_file or not os.path.exists(last_recorded_file):
        messagebox.showwarning("提示", "没有可回放的录音！")
        if playback_last_rec_button:
            playback_last_rec_button.config(state=tk.NORMAL)
        return

    # 播放开始前，把回放按钮禁用（变灰），防止用户一直点。
    if playback_last_rec_button:
        playback_last_rec_button.config(state=tk.DISABLED)

    # 创建并启动线程，目标函数是 play_and_reenable。
    threading.Thread(target=play_and_reenable, args=(last_recorded_file,), daemon=True).start()


def play_and_reenable(filename):
    """
    功能：这是一个在子线程里运行的函数，它先播放声音，播放完后再把按钮恢复可用。
    """
    play_audio_blocking(filename)  # 执行播放。

    # 播放完后，需要把按钮变回正常状态 (NORMAL)。
    # 注意：在子线程里不能直接修改界面控件，必须用 root.after 通知主线程去修改。
    if root and playback_last_rec_button:
        try:
            root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})
        except tk.TclError:
            pass  # 如果此时窗口已经关闭了，就忽略错误。


# --- 文件加载功能 ---
def load_file():
    """
    功能：点击“加载音频文件”按钮时触发，弹出一个文件选择框让用户选文件。
    """
    global selected_filepath, enhanced_filepath, file_label, status_label, play_enhanced_button

    # 弹出系统自带的文件选择对话框，只允许选 .wav 文件。
    filepath = filedialog.askopenfilename(
        title="请选择一个 .wav 音频文件",
        filetypes=[("WAV files", "*.wav")]
    )
    # 如果用户点了取消，filepath 就是空的，直接返回。
    if not filepath:
        return

    # 更新全局变量。
    selected_filepath = filepath
    enhanced_filepath = ""  # 加载新文件后，清空旧的增强文件路径。
    filename = os.path.basename(filepath)  # 获取文件名（不带路径）。

    # 更新界面文字。
    if file_label:
        file_label.config(text=f"已加载: {filename}")
    if status_label:
        status_label.config(text="状态: 文件加载成功")
    # 因为刚加载还没做增强处理，所以禁用“播放增强音频”按钮。
    if play_enhanced_button:
        play_enhanced_button.config(state=tk.DISABLED)


def play_sound(filepath):
    """
    功能：使用 Pygame 播放音频。
    特点：Pygame 的 mixer 模块是非阻塞的，点击播放后代码会继续往下走，
    所以不会卡住界面，适合用来播放背景音乐或长录音。
    """
    global status_label
    if not filepath:
        messagebox.showwarning("提示", "没有可播放的文件路径！")
        return
    if not os.path.exists(filepath):
        messagebox.showerror("错误", f"文件不存在: {filepath}")
        return
    try:
        # 确保 mixer 初始化了。
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(filepath)  # 加载音频文件。
        pygame.mixer.music.play()  # 开始播放。

        # 更新状态栏文字，告诉用户正在放哪个文件。
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


# --- 语音增强 (Task 1: NR算法) ---
def enhance_voice_nr():
    """
    功能：点击“语音增强”按钮时触发。调用降噪算法处理选中的文件。
    """
    global selected_filepath, enhanced_filepath, status_label, play_enhanced_button

    if not selected_filepath:
        messagebox.showwarning("提示", "请先加载一个音频文件！")
        return

    try:
        if status_label:
            status_label.config(text="状态: 正在进行 NR 降噪...请稍候...")
            root.update_idletasks()

        # 1. 使用 librosa 加载音频文件。
        # sr=16000 表示强制将采样率转换为 16kHz，这是我们项目的统一标准。
        y, sr = librosa.load(selected_filepath, sr=16000)

        # 2. 调用下面的 _process_nr 函数进行实际的降噪计算。
        enhanced_y = _process_nr(y, sr, noise_region='start')

        if enhanced_y is None:
            if status_label: status_label.config(text="状态: NR 降噪处理失败")
            return

        # 3. 构造输出文件的文件名。
        # 比如原文件是 test.wav，新文件就是 test_enhanced_nr.wav。
        base_dir = os.path.dirname(selected_filepath)
        base_name = os.path.basename(selected_filepath)
        name, ext = os.path.splitext(base_name)
        enhanced_filename = f"{name}_enhanced_nr{ext}"
        enhanced_filepath = os.path.join(base_dir, enhanced_filename)

        # 4. 使用 soundfile 保存处理后的音频数据。
        # subtype='PCM_16' 表示保存为 16位 整数格式，兼容性最好。
        sf.write(enhanced_filepath, enhanced_y, sr, subtype='PCM_16')

        if status_label:
            status_label.config(text=f"状态: NR 语音增强完成！已保存至 {enhanced_filename}")
        # 处理成功后，启用播放按钮。
        if play_enhanced_button:
            play_enhanced_button.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("处理错误", f"NR 语音增强失败: {e}")
        enhanced_filepath = ""
        if play_enhanced_button: play_enhanced_button.config(state=tk.DISABLED)


def _process_nr(y, sr, noise_region='start'):
    """
    功能：降噪的核心逻辑函数。
    算法：使用 noisereduce 库的 Stationary (平稳) 噪声抑制算法。
    原理：假设音频的开头一段（比如前1秒）是纯噪音，没有说话声。
          算法会分析这段噪音的频率特征（噪音指纹），然后从整段音频中
          减去符合这个指纹的声音成分。
    """
    global status_label, prop_decrease_slider, n_std_thresh_slider
    try:
        # 1. 提取噪声样本 (Noise Clip)。
        # 我们假设前 1.0 秒是纯背景噪声。
        noise_duration = 1.0
        noise_samples = int(noise_duration * sr)  # 计算1秒有多少个采样点。

        if len(y) < noise_samples:
            # 如果文件本身都不足1秒，那就只能拿整个文件当噪音样本了（没办法的办法）。
            noise_clip = y
        else:
            # 截取开头的数据作为噪音样本。
            noise_clip = y[:noise_samples]

        # 2. 从界面的滑块获取用户设定的参数。
        prop_decrease = float(prop_decrease_slider.get())  # 降噪强度 (0.0 到 1.0)。
        n_std_thresh_stat = float(n_std_thresh_slider.get())  # 阈值 (判断多大声音算噪音)。

        # 3. 调用 noisereduce 库的 reduce_noise 函数进行处理。
        enhanced_y = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_clip,  # 必须传入噪音样本。
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh_stat  # 传入平稳模式的参数。
        )
        return enhanced_y  # 返回降噪后的音频数据。
    except Exception as e:
        messagebox.showerror("降噪内部错误", f"_process_nr 失败: {e}")
        return None


# --- 说话人识别功能 (Task 2) ---
def start_register_thread():
    """GUI 回调：点击'注册声纹'按钮时触发。"""
    global speaker_name_entry
    speaker_name = speaker_name_entry.get()  # 获取用户在输入框里写的名字。
    if not speaker_name:
        messagebox.showwarning("提示", "请输入说话人姓名！")
        return
    # 禁用按钮，防止用户在注册过程中又点一次。
    register_button.config(state=tk.DISABLED)
    # 启动后台线程执行注册任务。
    threading.Thread(target=register_speaker, args=(speaker_name,), daemon=True).start()


def register_speaker(speaker_name):
    """
    后台任务：注册说话人。
    逻辑：
    1. 让用户连续录音 5 次，每次 4 秒。
    2. 对每次录音进行降噪。
    3. 提取 MFCC 特征。
    4. 将 5 次的特征拼接到一起。
    5. 用这些特征训练一个 GMM (高斯混合模型)。
    6. 把训练好的模型保存到硬盘。
    """
    global status_label, last_recorded_file, playback_last_rec_button
    all_features = np.asarray(())  # 初始化一个空数组，用来存所有特征。
    try:
        temp_dir = "./temp_recordings"
        model_dir = "./gmm_models"
        os.makedirs(temp_dir, exist_ok=True)  # 确保文件夹存在。
        os.makedirs(model_dir, exist_ok=True)

        record_count = 5  # 设定要录 5 次。

        # 循环 5 次。
        for i in range(record_count):
            if status_label:
                status_label.config(text=f"请为 {speaker_name} 录音 {i + 1}/{record_count} (4秒)...")
            time.sleep(0.5)  # 暂停 0.5 秒给用户准备。

            # 1. 录音
            raw_file = os.path.join(temp_dir, f"{speaker_name}_raw_{i + 1}.wav")
            recorded_path = record_audio(raw_file, 4)
            if recorded_path is None:
                # 如果录音失败，恢复按钮并退出。
                if root and register_button:
                    root.after(0, register_button.config, {"state": tk.NORMAL})
                return

            last_recorded_file = recorded_path
            # 允许回放刚才录的那一段。
            if playback_last_rec_button:
                root.after(0, playback_last_rec_button.config, {"state": tk.NORMAL})

            # 2. 降噪预处理
            # 加载录音。
            y, sr = librosa.load(recorded_path, sr=16000)
            # 降噪（这很重要，因为噪音会影响特征提取的准确性）。
            enhanced_y = _process_nr(y, sr, noise_region='start')

            if enhanced_y is None: continue

            # 3. 特征提取
            # 调用我们在 mfcc_coeff.py 里写的函数。
            vector = extract_features(enhanced_y, sr)

            if vector is None or vector.size == 0:
                print(f"警告：录音 {i + 1} 特征提取失败。")
                continue

            # 4. 特征拼接
            # 把这一次的特征堆叠到总特征数组里。
            if all_features.size == 0:
                all_features = vector
            else:
                if vector.shape[1:] != all_features.shape[1:]:
                    continue  # 如果维度不对，跳过这次。
                all_features = np.vstack((all_features, vector))

        # 检查是否收集到了足够的特征。
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
        # GMM 是一种统计模型，适合模拟声纹特征的分布。
        # n_components=16 表示用 16 个高斯分布来拟合。
        gmm = GaussianMixture(n_components=16,
                              covariance_type='diag',  # 使用对角协方差矩阵，计算更快。
                              max_iter=200,
                              random_state=0)
        gmm.fit(all_features)  # 开始训练。

        # 6. 保存模型文件。
        model_path = os.path.join(model_dir, f"{speaker_name}.gmm")
        joblib.dump(gmm, model_path)  # 保存成文件。

        if status_label:
            status_label.config(text=f"成功！ {speaker_name} 的声纹已注册。")

    except Exception as e:
        messagebox.showerror("注册失败", f"发生错误: {e}")
    finally:
        # 无论成功失败，都要恢复注册按钮。
        if root and register_button:
            try:
                root.after(0, register_button.config, {"state": tk.NORMAL})
            except tk.TclError:
                pass


def start_recognize_thread():
    """GUI 回调：点击'识别说话人'按钮。"""
    recognize_button.config(state=tk.DISABLED)
    threading.Thread(target=recognize_speaker, daemon=True).start()


def recognize_speaker():
    """
    后台任务：识别说话人。
    逻辑：
    1. 加载所有已注册的 GMM 模型。
    2. 录制一段新的音频 (5秒)。
    3. 对音频进行降噪和特征提取。
    4. 将特征分别输入每个模型，计算得分 (Score)。
    5. 得分最高的那个模型对应的名字，就是识别结果。
    """
    global status_label, last_recorded_file, playback_last_rec_button
    try:
        model_dir = "./gmm_models/"
        temp_dir = "./temp_recordings"
        os.makedirs(temp_dir, exist_ok=True)

        # 1. 扫描文件夹，加载所有 .gmm 模型文件。
        gmm_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.gmm')]
        if not gmm_files:
            if status_label: status_label.config(text="错误: 声纹库为空！")
            if root and recognize_button: root.after(0, recognize_button.config, {"state": tk.NORMAL})
            return

        models = []
        speakers = []
        for f in gmm_files:
            try:
                models.append(joblib.load(f))  # 加载模型。
                speakers.append(os.path.basename(f).split('.gmm')[0])  # 提取文件名作为人名。
            except Exception as e:
                print(f"警告：加载模型失败: {e}")

        # 2. 录音 5 秒。
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

        # 3. 预处理 (降噪)。
        if status_label: status_label.config(text="正在处理和识别...")
        y, sr = librosa.load(recorded_path, sr=16000)
        enhanced_y = _process_nr(y, sr, noise_region='start')
        if enhanced_y is None: return

        # 4. 提取特征。
        vector = extract_features(enhanced_y, sr)
        if vector is None or vector.size == 0:
            if status_label: status_label.config(text="识别失败：无法提取特征。")
            if root and recognize_button: root.after(0, recognize_button.config, {"state": tk.NORMAL})
            return

        # 5. 计算得分 (Log-Likelihood)。
        log_likelihood = np.zeros(len(models))  # 创建一个数组存得分。
        for i in range(len(models)):
            gmm = models[i]
            # score_samples 计算每个特征点属于该模型的概率对数。
            scores = gmm.score_samples(vector)
            # sum() 把所有点的得分加起来，作为整段音频的总得分。
            log_likelihood[i] = scores.sum()

        # 6. 判决。
        # np.nanargmax 找出得分最高的那个索引。
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


# --- 语音转文字 (Task 3: Baidu API) ---
def recognize_speech_baidu(file_path):
    """
    核心功能：调用百度 API 进行语音识别。
    逻辑：读取本地文件 -> 发送 POST 请求给百度 -> 接收 JSON 结果 -> 解析出文字。
    """
    global client, status_label
    if client is None:  # 如果客户端没初始化成功，直接报错。
        messagebox.showerror("API 错误", "Baidu API 未成功初始化！")
        return None

    try:
        # 以二进制模式读取音频文件数据。
        with open(file_path, 'rb') as f:
            audio_data = f.read()

        # 调用 SDK 的 asr 方法发送请求。
        # 'wav': 告诉百度这是 WAV 格式。
        # 16000: 告诉百度采样率是 16k。
        # dev_pid=1537: 这是百度普通话输入法模型的编号，识别中文最准。
        result = client.asr(audio_data, 'wav', 16000, {'dev_pid': 1537, })

        # 检查返回结果里的 err_no，如果是 0 表示成功。
        if result and result.get('err_no') == 0:
            # result['result'] 是一个列表，里面是识别出的句子，我们用空格把它拼起来。
            return " ".join(result.get('result', []))
        else:
            # 如果失败，获取错误信息。
            error_msg = result.get('err_msg', '未知错误')
            messagebox.showerror("API 识别失败", f"错误信息: {error_msg}")
            return None
    except Exception as e:
        messagebox.showerror("API 调用失败", f"异常: {e}")
        return None


def start_speech_to_text_thread():
    """
    GUI 回调：语音转文字按钮。
    逻辑：这是一个“开关”按钮。
    - 如果正在录音，点击它意味着“停止录音并识别”。
    - 如果没在录音，点击它意味着“开始录音”。
    """
    global stt_thread, stt_button, stt_stop_event, status_label, stt_result_text

    # 检查线程是否活着（是否正在录音）。
    if stt_thread and stt_thread.is_alive():
        # 如果在录音，更新状态，并发出停止信号。
        if status_label: status_label.config(text="录音停止... 正在处理...")
        stt_stop_event.set()  # 设置信号为 True，通知子线程停止循环。
        if stt_button: stt_button.config(text="正在处理...", state=tk.DISABLED)
    else:
        # 如果没在录音，开始录音流程。
        if client is None:  # 先检查 API 是否配置好。
            recognize_speech_baidu(None)
            return

        stt_stop_event.clear()  # 重置信号灯为 False。
        # 清空文本框里的旧内容。
        if stt_result_text:
            stt_result_text.config(state=tk.NORMAL)
            stt_result_text.delete('1.0', tk.END)
            stt_result_text.config(state=tk.DISABLED)

        # 启动录音子线程。
        stt_thread = threading.Thread(target=recognize_speech_task, daemon=True)
        stt_thread.start()
        # 更新按钮文字，变成红色提醒用户正在录音。
        if stt_button:
            stt_button.config(text="正在录音... (点击停止)", state=tk.NORMAL, fg="red")


def recognize_speech_task():
    """
    后台任务：STT 的完整工作流。
    流式录音 -> 保存文件 -> 降噪 -> 上传百度 -> 显示结果。
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

        # 循环读取录音数据，直到主线程把 stt_stop_event 设置为 True。
        while not stt_stop_event.is_set():
            try:
                # exception_on_overflow=False 防止缓冲区溢出导致程序崩溃。
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError:
                pass

        # 停止录音，关闭流。
        try:
            stream.stop_stream();
            stream.close();
            p.terminate()
        except:
            pass

        if len(frames) == 0: return  # 如果没录到东西，直接退出。

        # 1. 保存原始录音文件。
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

        # 2. 降噪处理。
        # 因为云端识别对噪音很敏感，本地先降噪一次可以大大提高识别准确率。
        y, sr = librosa.load(test_file, sr=16000)
        enhanced_y = _process_nr(y, sr, noise_region='start')

        audio_to_recognize = test_file  # 默认使用原始文件。
        if enhanced_y is not None:
            enhanced_file = os.path.join(temp_dir, "stt_test_enhanced.wav")
            try:
                sf.write(enhanced_file, enhanced_y, sr, subtype='PCM_16')
                audio_to_recognize = enhanced_file  # 如果降噪成功，就使用降噪后的文件上传。
            except:
                pass

        if status_label: status_label.config(text="正在调用 Baidu API 进行语音识别...")

        # 3. 调用 API 获取结果。
        text_result = recognize_speech_baidu(audio_to_recognize)

        # 4. 显示结果。
        if text_result:
            if status_label: status_label.config(text="状态: 语音转文字成功！")

            # 定义一个小函数来更新 GUI 里的文本框，因为要在主线程里运行。
            def update_stt_text():
                stt_result_text.config(state=tk.NORMAL)  # 先设为可编辑。
                stt_result_text.delete('1.0', tk.END)  # 清空。
                stt_result_text.insert(tk.END, text_result)  # 插入文字。
                stt_result_text.config(state=tk.DISABLED)  # 再设为不可编辑（只读）。

            try:
                root.after(0, update_stt_text)  # 通知主线程执行更新。
            except:
                pass
        else:
            if status_label: status_label.config(text="状态: 语音转文字失败。")

    except Exception as e:
        if status_label: status_label.config(text="状态: 语音转文字异常终止")
    finally:
        # 清理工作：把线程变量置空，恢复按钮状态。
        stt_thread = None
        if root and stt_button:
            try:
                root.after(0, stt_button.config,
                           {"text": "8. 语音转文字 (点击开始)", "state": tk.NORMAL, "fg": "purple"})
            except:
                pass


# --- 语音情感识别 (Task 4: Hugging Face / HuBERT) ---

def get_emotion_classifier():
    """
    功能：加载情感识别模型。
    策略：懒加载 (Lazy Loading)。即程序启动时不加载（为了启动快），
    第一次点击按钮时才加载，并把加载好的模型存在内存里 (loaded_models)，
    第二次点击时直接用内存里的，不用再读文件了。
    """
    global loaded_models, loaded_processors, status_label
    model_id = SER_MODEL_ID

    # 1. 检查内存缓存里有没有。
    if model_id in loaded_models and model_id in loaded_processors:
        return loaded_models[model_id], loaded_processors[model_id]

    if status_label:
        status_label.config(text=f"首次加载情感识别模型 ({model_id}), 请稍候...")
        root.update_idletasks()

    try:
        if not TRANSFORMERS_AVAILABLE:
            messagebox.showerror("库错误", "未找到 Transformers 库。")
            return None, None

        # 2. 加载处理器 (Wav2Vec2FeatureExtractor)。
        # 它的作用是把声音波形变成模型能“吃”进去的数字格式。
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        # 3. 加载模型配置和权重。
        # 使用我们自定义的 HubertForSpeechClassification 类。
        config = AutoConfig.from_pretrained(model_id)
        model = HubertForSpeechClassification.from_pretrained(model_id, config=config)

        # 4. 尝试使用 GPU 加速。
        # 如果电脑有显卡 (CUDA)，就用显卡跑，速度会快几十倍；否则用 CPU。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # 把模型搬到 GPU 上。
        model.eval()  # 设置为评估模式 (Evaluation Mode)，告诉模型现在不训练，只做推理。

        # 5. 存入缓存字典。
        loaded_models[model_id] = (model, device)
        loaded_processors[model_id] = processor

        if status_label: status_label.config(text=f"SER 模型加载成功!")
        return loaded_models[model_id], loaded_processors[model_id]

    except Exception as e:
        messagebox.showerror("模型加载失败", str(e))
        return None, None


def start_emotion_recognition_thread():
    """GUI 回调：点击'识别情感'按钮。"""
    if not TRANSFORMERS_AVAILABLE:
        messagebox.showerror("库错误", "库未安装")
        return
    emotion_button.config(state=tk.DISABLED)
    # 启动后台线程。
    threading.Thread(target=recognize_emotion_task, daemon=True).start()


def recognize_emotion_task():
    """
    后台任务：情感识别完整流程。
    1. 获取/加载模型。
    2. 录音 4 秒。
    3. 降噪。
    4. 音频预处理（去静音、归一化、转张量）。
    5. 模型推理。
    6. 结果解析和排序。
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

        # 3. 降噪 (推荐步骤)
        audio_input_path = recorded_path
        if status_label: status_label.config(text="正在降噪...")

        y, sr = librosa.load(recorded_path, sr=16000)
        enhanced_y = _process_nr(y, sr, noise_region='start')

        if enhanced_y is not None:
            enhanced_file_path = os.path.join(temp_dir, "emotion_test_enhanced.wav")
            sf.write(enhanced_file_path, enhanced_y, sr, subtype='PCM_16')
            audio_input_path = enhanced_file_path  # 使用降噪后的文件进行识别。

        if status_label: status_label.config(text=f"正在推理...")

        # --- 关键：深度学习推理 ---
        try:
            # 加载音频到内存。
            audio_array, sampling_rate = librosa.load(audio_input_path, sr=16000)

            # [优化 A] 去除首尾静音 (Trim)。
            # 把说话前后没有声音的部分切掉，防止干扰模型。
            audio_array, _ = librosa.effects.trim(audio_array, top_db=25)

            # [优化 B] 长度检查。
            if len(audio_array) < 0.5 * 16000:  # 如果切完不到 0.5 秒，模型可能会报错。
                if status_label: status_label.config(text="错误：有效语音太短")
                if root and emotion_button: root.after(0, emotion_button.config, {"state": tk.NORMAL})
                return

            # [优化 C] 音量归一化 (Normalize)。
            # 把声音从小声放大到标准音量。
            audio_array = librosa.util.normalize(audio_array)

            # 预处理：将 numpy 数组转换为 PyTorch 张量 (Tensor)。
            inputs_dict = processor(
                audio_array,
                sampling_rate=16000,
                padding=True,
                truncation=True,
                max_length=6 * 16000,  # 限制最大长度。
                return_tensors="pt"  # 返回 PyTorch 张量。
            )

            # 把数据移到 GPU (如果可用)。
            input_values = inputs_dict.input_values.to(device)

            # 模型推理 (Forward Pass)。
            with torch.no_grad():  # 不计算梯度，节省内存。
                logits = model(input_values)  # 得到模型的原始输出 (Logits)。

            # Softmax 归一化：把 Logits 转换成概率 (0到1之间)。
            probabilities = F.softmax(logits, dim=1)
            scores = probabilities[0].cpu().numpy()  # 把结果转回 CPU，变成 numpy 数组方便处理。

            # 标签映射字典 (0->anger, 1->fear...)。
            id2label = {0: "anger", 1: "fear", 2: "happy", 3: "neutral", 4: "sad", 5: "surprise"}

            # 整理结果。
            results_list = []
            for i in range(len(scores)):
                results_list.append({"label": id2label[i], "score": float(scores[i])})

            # 按置信度从高到低排序。
            results_list.sort(key=lambda x: x["score"], reverse=True)

            # 拼接显示文字。
            best_emotion = results_list[0]
            display_text = f"识别结果: {best_emotion['label']} (置信度: {best_emotion['score']:.2f})\n\n详细结果:\n"
            for res in results_list:
                display_text += f"  - {res['label']}: {res['score']:.2f}\n"

            # 更新界面。
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
        # 恢复按钮状态。
        if root and emotion_button:
            try:
                root.after(0, emotion_button.config, {"state": tk.NORMAL})
            except:
                pass


# --- GUI 界面创建 (主入口) ---
pygame.mixer.init()  # 初始化 Pygame 音频混音器。
root = tk.Tk()  # 创建 Tkinter 主窗口。
root.title("数字音效处理系统 V4.9 (原始 noisereduce)")  # 设置窗口标题。
root.geometry("550x750")  # 设置窗口大小。

# 创建主容器 Frame。
main_frame = tk.Frame(root, padx=10, pady=10)
main_frame.pack(fill=tk.BOTH, expand=True)  # 让容器填满窗口。

# 标题标签。
title_label = tk.Label(main_frame, text="数字音效处理系统", font=("Arial", 16, "bold"))
title_label.pack(pady=(0, 10))  # pady 设置垂直间距。

# --- 状态显示区 ---
status_frame = tk.Frame(main_frame)
status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

file_label = tk.Label(status_frame, text="已加载: (无)", font=("Arial", 10), wraplength=500, anchor='w')
file_label.pack(fill=tk.X)

status_label = tk.Label(status_frame, text="状态: 准备就绪", font=("Arial", 10, "italic"), fg="blue", anchor='w')
status_label.pack(fill=tk.X, pady=(5, 0))

# --- Tab 选项卡控件 ---
# Notebook 是 Tkinter 里的选项卡控件。
tab_control = ttk.Notebook(main_frame)
tab1 = ttk.Frame(tab_control, padding=10)  # 创建页面1，用于文件处理。
tab2 = ttk.Frame(tab_control, padding=10)  # 创建页面2，用于实时处理。
tab_control.add(tab1, text='文件处理 (任务1)')  # 添加页面1。
tab_control.add(tab2, text='实时处理 (任务2, 3, 4)')  # 添加页面2。
tab_control.pack(expand=True, fill='both')

# --- 任务1: 语音增强 GUI (Tab 1) ---
file_frame = tk.Frame(tab1, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
file_frame.pack(fill=tk.X, pady=5)
task1_label = tk.Label(file_frame, text="任务1: 语音增强 (处理文件)", font=("Arial", 12, "bold"))
task1_label.pack(anchor="w")


# [终极修正版] Task 1 现场录音逻辑
# 这个函数专门给 Tab 1 里的录音按钮用。
def record_for_task1():
    """
    Task 1 的现场录音按钮逻辑。
    特殊处理：在录音前必须先强制停止播放并释放文件锁。
    """
    global selected_filepath, file_label, status_label, play_enhanced_button

    temp_dir = "./temp_recordings"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    filename = os.path.join(temp_dir, "live_record_enhancement.wav")

    # 关键：释放文件锁，防止报错 "WinError 32"。
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()  # 停止播放。
            try:
                pygame.mixer.music.unload()  # 卸载文件 (Pygame 2.0+)。
            except AttributeError:
                # 如果版本太旧，重启 mixer。
                pygame.mixer.quit();
                pygame.mixer.init()
    except Exception as e:
        print(e)
    time.sleep(0.1)  # 等待操作系统释放文件句柄。

    # 开始录音。
    if status_label: status_label.config(text="正在现场录音 (8秒)...")
    root.update_idletasks()
    filepath = record_audio(filename, 8)

    # 录音成功后，自动把录好的文件设为“选中文件”。
    if filepath:
        selected_filepath = filepath
        global enhanced_filepath
        enhanced_filepath = ""
        if file_label: file_label.config(text=f"已现场采集: {os.path.basename(filename)}")
        if status_label: status_label.config(text="录音完成！请点击增强。")
        if play_enhanced_button: play_enhanced_button.config(state=tk.DISABLED)
    else:
        if status_label: status_label.config(text="录音失败")


# --- Tab 1 控件布局 ---
# 录音按钮
record_task1_button = tk.Button(file_frame, text="0. 现场采集音频 (录音 8s)", command=record_for_task1,
                                font=("Arial", 12, "bold"), fg="red")
record_task1_button.pack(fill=tk.X, pady=(5, 0))

# 加载文件按钮
load_button = tk.Button(file_frame, text="1. 加载音频文件 (.wav)", command=load_file, font=("Arial", 12))
load_button.pack(fill=tk.X, pady=(5, 0))

# 播放原始文件按钮
play_original_button = tk.Button(file_frame, text="2. 播放原始音频", command=lambda: play_sound(selected_filepath),
                                 font=("Arial", 12))
play_original_button.pack(fill=tk.X, pady=5)

# 参数滑块区域
processing_frame = tk.Frame(file_frame)
processing_frame.pack(fill=tk.X, pady=5)

# 滑块1：降噪强度
slider_frame_1 = tk.Frame(processing_frame)
slider_frame_1.pack(fill=tk.X, padx=5, pady=2)
tk.Label(slider_frame_1, text="降噪强度 (0-1):", font=("Arial", 10)).pack(side=tk.LEFT)
prop_decrease_slider = tk.Scale(slider_frame_1, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
prop_decrease_slider.set(1.0)  # 默认值 1.0
prop_decrease_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

# 滑块2：噪声阈值
slider_frame_2 = tk.Frame(processing_frame)
slider_frame_2.pack(fill=tk.X, padx=5, pady=2)
tk.Label(slider_frame_2, text="噪声阈值 (0-3):", font=("Arial", 10)).pack(side=tk.LEFT)
n_std_thresh_slider = tk.Scale(slider_frame_2, from_=0.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL)
n_std_thresh_slider.set(1.0)  # 默认值 1.0
n_std_thresh_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

# 执行增强按钮
enhance_button = tk.Button(processing_frame, text="3. 语音增强 (本地降噪)", command=enhance_voice_nr,
                           font=("Arial", 12, "bold"), fg="blue")
enhance_button.pack(fill=tk.X, pady=5)

# 播放增强后按钮
play_enhanced_button = tk.Button(processing_frame, text="4. 播放增强后音频",
                                 command=lambda: play_sound(enhanced_filepath), font=("Arial", 12), state=tk.DISABLED)
play_enhanced_button.pack(fill=tk.X, pady=5)

# --- 任务2: 说话人识别 GUI (Tab 2) ---
speaker_rec_frame = tk.Frame(tab2, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
speaker_rec_frame.pack(fill=tk.X, pady=10)
task2_label = tk.Label(speaker_rec_frame, text="任务2: 说话人识别 (GMM)", font=("Arial", 12, "bold"))
task2_label.pack(anchor="w")

register_frame = tk.Frame(speaker_rec_frame)
register_frame.pack(fill=tk.X, pady=5)
tk.Label(register_frame, text="说话人姓名:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(5, 5))
speaker_name_entry = tk.Entry(register_frame, font=("Arial", 10))
speaker_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

# 注册按钮
register_button = tk.Button(register_frame, text="5. 注册声纹 (录5次*4s)", command=start_register_thread,
                            font=("Arial", 12, "bold"), fg="green")
register_button.pack(side=tk.LEFT, padx=5)

# 识别按钮
recognize_button = tk.Button(speaker_rec_frame, text="6. 识别说话人 (录 5 秒)", command=start_recognize_thread,
                             font=("Arial", 12, "bold"), fg="red")
recognize_button.pack(fill=tk.X, pady=(10, 5))

# 回放按钮
playback_last_rec_button = tk.Button(speaker_rec_frame, text="7. 回放上次录音", command=start_playback_thread,
                                     font=("Arial", 12), state=tk.DISABLED)
playback_last_rec_button.pack(fill=tk.X, pady=5)

# --- 任务3 语音转文字 GUI (Tab 2) ---
stt_frame = tk.Frame(tab2, relief=tk.GROOVE, borderwidth=2, padx=5, pady=5)
stt_frame.pack(fill=tk.X, pady=10)
task3_label = tk.Label(stt_frame, text="任务3: 语音转文字 (Baidu API)", font=("Arial", 12, "bold"))
task3_label.pack(anchor="w")

stt_button = tk.Button(stt_frame, text="8. 语音转文字 (点击开始/停止)", command=start_speech_to_text_thread,
                       font=("Arial", 12, "bold"), fg="purple")
stt_button.pack(fill=tk.X, pady=5)

stt_result_text = tk.Text(stt_frame, height=3, font=("Arial", 10), state=tk.DISABLED, wrap=tk.WORD)
stt_result_text.pack(fill=tk.X, pady=5)

# --- 任务4 语音情感识别 GUI (Tab 2) ---
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
    """
    功能：当用户点击窗口右上角的 X 关闭程序时触发。
    """
    print("正在关闭应用程序...")
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()  # 关闭 Pygame 混音器。
    except Exception:
        pass
    try:
        root.destroy()  # 销毁主窗口，结束程序。
    except tk.TclError:
        pass


# --- 程序入口 ---
if __name__ == "__main__":
    # 只有直接运行这个文件时，才会执行下面的代码。
    root.protocol("WM_DELETE_WINDOW", on_closing)  # 绑定关闭窗口事件。
    root.mainloop()  # 进入 GUI 的主事件循环，程序开始运行并等待用户操作。