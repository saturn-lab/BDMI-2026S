## 20260428 BDMI课程小记
#### by 物理42 俞善斌
第十次课。

---

### 1. 深度学习与 Transformer 回顾

#### 深度学习基本流程
- 本周课件继续用深度学习概述作为铺垫，回顾了从线性分类器到深度人工神经网络的主线。
- 线性分类器和 SVM 可用于图像分类，但在复杂任务中表达能力有限。
- 深度神经网络训练仍围绕几件事展开：
  - 数据预处理
  - 网络结构和损失函数设计
  - 梯度下降
  - 学习率选取
  - 正则化解决过拟合
- 学习率策略包括指数衰减、\(1/t\) 衰减、分步衰减，也可以使用 RMSProp、Adam 等优化器。
- 正则化方法包括 L2 参数正则化、数据增强、Dropout 和 Ensemble Learning。

#### Transformer 结构
- Transformer 是一种 encoder-decoder 结构，核心优势是可以并行训练。
- 主要术语：
  - 自注意力（Self Attention）
  - 可缩放点积注意力（Scaled Dot-Product Attention）
  - 位置编码（Positional Encoding）
  - 遮挡（Masking）
  - 多头注意力（Multi-Head Attention）
- 编码器：多头自注意力层 + 全连接前馈网络。
- 解码器：遮挡多头自注意力层 + 编码器-解码器注意力层 + 前馈网络。
- Transformer 本身不能利用词序信息，需要加入位置 Embedding。
- Transformer 变体：
  - 类 GPT：自动回归 Transformer 模型
  - 类 BERT：自动编码 Transformer 模型
  - 类 BART/T5：序列到序列 Transformer 模型

---

### 2. 自动语音识别（ASR）基础

#### 基本概念
- 自动语音识别（Automatic Speech Recognition, ASR）：用计算机程序实现语音识别。
- 大词汇量连续语音识别系统（Large Vocabulary Continuous Speech Recognition, LVCSR）强调：
  - 大词汇量
  - 连续语音
- 语音信号是声波，也是时变信号。
- 语音信号的表示：
  - 波形图（Waveform）
  - 频率图（Frequency）
  - 时频谱图（Spectrogram）

#### 评价指标：WER
- 词错误率（Word Error Rate, WER）用于评价识别结果。
- 计算方式：将标准答案和识别结果对齐，错误率等于替换、插入、删除的总数除以标准答案长度。
  \[
  WER = \frac{S + I + D}{N}
  \]
  其中 \(S\) 为替换，\(I\) 为插入，\(D\) 为删除，\(N\) 为标准答案词数。
- 对齐应选择使 WER 最小的方式。
- 示例：
  - 标准答案：`my name is Andy Chen`
  - 识别结果：`my nick name are Andy Chen`
  - 插入、替换、删除共 3 个错误，WER = \(3/5 = 60\%\)
- WER 可以高于 1，例如标准答案为 `recognize speech`，识别结果为 `wreck a nice beach`，错误率可达 \(4/2 = 200\%\)。

#### 传统 ASR 方程
- 对给定语音信号 \(X\) 和单词序列 \(W\)，最优识别结果可表示为：
  \[
  W^* = \arg\max_W P(W|X)
  \]
- 应用贝叶斯公式后，传统路线可以分解为：
  - 声学模型（Acoustic Model）
  - 语言模型（Language Model）
- 传统语音识别架构：
  - 语音信号经过 STFT（短时傅里叶变换）
  - 连续语音分解成短期向量序列
  - 向量序列变换为音素序列
  - 音素序列变换为字母序列
  - 字母序列变换为词汇序列
- 常见传统技术：
  - HMM（Hidden Markov Model）
  - GMM（Gaussian Mixture Model）
  - Viterbi search algorithms
  - n-gram language models
  - MFCC（mel-frequency cepstral coefficients）

---

### 3. 深度学习语音识别与 CTC

#### 深度学习 ASR
- 深度学习模型需要训练数据和评价驱动的方法进行参数优化。
- 传统混合模型依赖 HMM、上下文相关 phone models、Viterbi 搜索、n-gram 语言模型等模块，模型较复杂，也依赖较多专业知识。
- 端到端语音识别模型（End-to-End）希望直接用神经网络连接输入语音和输出文本。
- 端到端语音识别流水线仍可分成三个主要部分：
  - 特征提取：原始音频 \(\rightarrow\) 原始波形、频谱图、时频谱图等特征
  - 声学模型：特征序列 \(\rightarrow\) 字符或音素序列概率
  - 语言模型：在候选转录中搜索最可能的 transcript

#### RNN 与 CLDNN
- RNN 可用于语音识别，因为语音是序列数据。
- Google CLDNN 结合 CNN、LSTM 和 DNN：
  - 对输入信号做时间域卷积
  - 再做频率域卷积，以减少频谱变化
  - 经过三层 LSTM
  - 最后通过一层 DNN
- 输入数据是以时间为下标的连续向量。

#### CTC 要解决的问题
- CTC（Connectionist Temporal Classification，连接主义时间分类）用于处理输入语音 \(X\) 和输出文字 \(Y\) 的对齐问题。
- ASR 中 \(X\) 和 \(Y\) 都是变长的，且没有确定的对齐方式。
- CTC 是 alignment-free 方法：
  - 对给定输入语音 \(X\)，给出所有可能文字序列 \(Y\) 的分布
  - 可以推断最可能输出，也可以计算给定输出序列 \(Y\) 的概率
- 通过 RNN 后形成概率分布矩阵，可生成多个识别 sequence。
- CTC Collapsing：不同路径经过折叠后可以得到相同输出。

#### CTC 损失与推断
- 对单个 \((X,Y)\) 对，CTC 目标函数依赖条件概率 \(p(Y|X)\)，该概率需要可微，才能使用梯度下降训练。
- CTC-loss 计算量可能很大，因为存在大量可能对齐。
- 课件示例中，当 \(T=100, U=50\) 时，所有可能路径数量巨大。
- 解决方法：用动态规划快速计算损失函数。
- CTC 推断方法：
  - Greedy decoding：每个时刻取概率最大输出，缺点是可能遗漏总概率更大的对齐
  - Beam search：保留多个候选路径，解决多个对齐合并后概率更大的情况
- 开源实现包括百度研究院 `warp-ctc`，TensorFlow2 中也提供 CTC loss 和 beam search 接口。

---

### 4. DeepSpeech 与中文语音识别

#### DeepSpeech 1
- DeepSpeech 是百度提出的端到端语音识别系统示例。
- DeepSpeech 1 有 5 个隐藏层，其中第 4 层为双向 LSTM。

#### DeepSpeech2
- DeepSpeech2 输入：时频谱图。
- 英文输出集合：
  \[
  \{a,b,c,\ldots,z,space,apostrophe,blank\}
  \]
- 中文输出集合：包含罗马字母表和约 6000 个汉字。
- 模型组成：
  - 卷积层（Conv. Layer）
  - 循环层（Recurrent Layer）
  - 全连接层（FC Layer）
- 优化中需要控制语言模型和 CTC 网络的贡献比率。

#### 实现技巧
- BatchNorm：批归一化，加速收敛。
- SortaGrad / SortaGrid：短句优先训练，保证 CTC 平稳性。
- GRU：GRU 和 LSTM 准确性相差不大，但 GRU 运算更快。
- 单向模型 + Lookahead Convolution：因为双向 LSTM 的时延达不到要求，所以用未来上下文为 2 的单向层做折中。
- 中文语音识别系统开发还需要中文语料准备、端到端模型、GPU 和智能硬件开发板等计算资源。

---

### 5. Whisper 自动语音识别

#### Whisper 基本信息
- Whisper 是 OpenAI 的自动语音识别系统。
- 训练数据：从网络收集的 68 万小时多语言、多任务监督数据。
- 语言规模：98 种语言。
- 课件中强调 Whisper 的语音识别能力达到人类水准。

#### 模型输入与任务格式
- Whisper 使用 Transformer Seq2Seq 模型。
- 输入音频处理流程：
  - 音频分割成 30 秒小段
  - 转换为 log-Mel 频谱图
  - 输入编码器
- 解码器预测对应文本，并与特殊标记混合。
- 特殊标记用于指导单个模型完成：
  - 语言识别
  - 短语级时间戳
  - 多语言语音转录
  - 语音翻译
  - 语音活动检测
- 多任务训练格式将不同语音处理任务统一表示为解码器要预测的 token 序列。

#### 训练设置与使用
- 训练采用跨加速器数据并行和 FP16 混合精度。
- 集成动态损失缩放和激活检查点，保证效率与稳定性。
- 优化器：AdamW。
- 使用梯度范数裁剪。
- 学习率经过 2048 次更新预热后线性衰减到 0。
- 批大小为 256 个样本段。
- 课件提到训练约 220 次权重更新，约 2-3 个完整数据周期。
- Whisper 有五种模型，其中四种只有英文版本，提供速度和准确性的折中。
- 可通过命令行、Python、HuggingFace Transformers.js 和 Buzz 工具使用。

---

### 6. PyTorch 音频识别实战

#### 音频识别任务
- 音频识别：用深度学习模型分析音频信号，识别其中的语义内容。
- 技术本质：将时域信号转换为特征表示，再通过神经网络进行模式分类。
- 应用场景：
  - 语音助手
  - 声纹识别
  - 音乐分类
  - 声音事件检测

#### torchaudio
- `torchaudio` 是 PyTorch 的音频处理核心库。
- 功能包括：
  - `load()`：加载音频
  - `save()`：保存音频
  - `resample()`：重采样
  - `spectrogram()`：生成频谱图
  - `mfcc()`：提取梅尔频谱特征
  - 内置常用音频数据集
- 支持 WAV、FLAC、MP3、OGG 等格式。

#### 音频预处理流程
- 加载音频，读取波形数据、采样率和通道数。
- 统一采样率，常用 16kHz 或 22.05kHz。
- 降噪处理，例如频谱减法。
- 分帧加窗，常用汉明窗或汉宁窗。
- 提取特征，计算 Mel Spectrogram、MFCC 或梅尔滤波器组。
- 最终输出可供 CNN 处理的频谱图图像。

#### 简单 CNN 分类结构
- 输入：Mel Spectrogram，形状为 `(batch, 1, freq, time)`。
- 示例结构：
  - `Conv2d(1, 32, 3x3)` + ReLU + MaxPool
  - `Conv2d(32, 64, 3x3)` + ReLU + MaxPool
  - 展平层
  - 全连接层
  - Dropout 防止过拟合
  - 输出层为 `num_classes`
- 可替换为 ResNet 等预训练模型提升效果。

#### 训练与推理
- 训练三要素：
  - 损失函数：`CrossEntropyLoss`
  - 优化器：Adam
  - 学习率：如 0.001
- 训练循环：前向传播、计算损失、反向传播、优化器更新、打印 epoch 统计。
- 推理时使用：
  - `model.eval()`
  - `torch.no_grad()`
- 常见问题与优化：
  - OOM：限制最大长度，使用动态 padding
  - 采样率不匹配：统一 resample
  - 类别不平衡：WeightedRandomSampler 或数据增强
  - 数据增强：Time Stretch、Pitch Shift、添加噪声
  - 预训练模型：VGGish、Wav2Vec2 等

#### 课件脚本内容
- `download_audio_datasets.py` 支持下载：
  - SpeechCommands：约 105,000 个 1 秒音频片段，35 个语音命令类别，16kHz 单声道
  - UrbanSound8K：8,732 个城市声音片段，10 个类别
  - ESC-50：2,000 个 5 秒环境声音片段，50 个类别
- `音频识别.py` 的实现流程：
  - 从 `audio_data` 按类别目录读取 `.wav` / `.mp3`
  - 用 `soundfile` 读取音频，避开 `torchaudio` 加载问题
  - 重采样到 16kHz
  - 转单声道
  - 裁剪或 padding 到 3 秒
  - 转 Mel Spectrogram 并取 log
  - 做标准化
  - 用 3 层 CNN + Dropout + Linear 分类
  - 保存最佳模型到 `checkpoints/best_model.pth`