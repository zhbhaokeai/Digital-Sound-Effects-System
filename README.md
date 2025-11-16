# 数字音效处理系统 (Digital Sound Effects System)

**课程名称**：数字信号处理B (Digital Signal Processing B)  
**项目类型**：课程设计项目 (Course Project)  
**开发语言**：Python 3.9+  
**GUI 框架**：Tkinter

---

## 📖 项目简介

本项目是一个基于 Python 的综合性数字语音处理平台。系统集成了多种数字信号处理 (DSP) 算法与现代深度学习模型，旨在实现对语音信号的增强、分析与识别。

用户可以通过图形化界面 (GUI) 方便地进行现场录音或加载本地文件，并直观地查看处理结果。

## ✨ 主要功能

1.  **语音增强 (Speech Enhancement)**
    * 利用谱减法与平稳噪声抑制算法 (Spectral Gating)，有效去除环境背景噪声。
    * 支持手动调节降噪强度与阈值，实时试听增强效果。

2.  **说话人识别 (Speaker Recognition)**
    * 基于 MFCC 特征提取与 GMM (高斯混合模型) 的声纹识别系统。
    * 支持用户注册声纹（5次循环录音机制）与身份验证。

3.  **语音情感识别 (Speech Emotion Recognition)**
    * 集成 HuggingFace 的 HuBERT 预训练模型 (Transformer 架构)。
    * 能够识别 6 种情感：愤怒 (Anger)、恐惧 (Fear)、快乐 (Happy)、中性 (Neutral)、悲伤 (Sad)、惊讶 (Surprise)。

4.  **智能语音转文字 (Speech-to-Text)**
    * 采用“本地降噪预处理 + 百度 AI 云端识别”的端云结合架构。
    * 支持高精度的中文语音听写。

---

## 🛠️ 环境安装 (Installation)

为了确保项目能正常运行，请按照以下步骤配置环境。

### 1. 克隆仓库
```bash
git clone [https://github.com/你的用户名/Digital-Sound-Effects-System.git](https://github.com/你的用户名/Digital-Sound-Effects-System.git)
cd Digital-Sound-Effects-System

2. 安装依赖库
请确保你的电脑安装了 Python (建议 3.8 或以上版本)。在 PyCharm 的终端 (Terminal) 运行：
pip install -r requirements.txt