## 自动语音识别（ASR，automatic speech recognition）

### 基本原理

语音信号映射为文字序列

### 挑战

发音、词汇量、环境

### 语音的数字化

- 波形图
- 频谱图(FFT)
- STFT得到时频谱图

### 评价指标

词错误率（word error rate）

$WER=(S+D+I)/N$

- S 替换
- I 插入
- D 删除
- N 标准答案词数

### 经典方法

#### 概率图模型

给定语音信号X，找到最有可能的词序列W*

$W^*=argmax\ P(W|X) = argmax\ P(X|W)P(W)$

- **声学模型**：
  - 给定词W，听起来像特征X的概率
  - GMM-HMM音素建模
- **语言模型**：
  - 词序列W本身存在的先验概率
  - N-gram词语搭配

#### 混合模型架构

- 2010年DNN替代高斯混合模型（GMM）

- 整体状态转移逻辑仍然遵循隐藏马尔可夫模型（HMM）

- 仍然依赖专家知识，需要复杂的特征工程，无法实现全系统联合优化

#### 端到端模型

将声学模型、发音词典和语言模型统一在一个网络中联合优化

**核心挑战**：语音和文字序列的对齐问题

### CTC

对齐无关（alignment-free），直接计算所有可能路径的概率总和

#### 原理

1. 引入blank标记
   - 扩展字符集，引入空白标记<b>
   - RNN的每个时间步输出包含<b>的字符概率分布
2. 路径折叠
   - 合并连续的相同字符
   - 移除所有<b>标记

#### CTC训练与推断

1. 训练：
   - CTC Loss，最大化所有能折叠成正确文本的路径概率之和
   - 用DP降低计算开销
2. 解码：
   - 贪心搜索
   - 束搜索

#### DeepSpeech2架构解析

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260428141243418.png" alt="image-20260428141243418" style="zoom:40%;" />



## PyTorch语音识别

### 音频预处理

1. **加载音频**：

   ```py
   torchaudio.load()
   ```

2. **重采样**：常用16kHz或22.05kHz

3. **降噪处理**：频谱减法移除背景噪音

4. **分帧加窗**：hamming/hanning切分为短帧保留时域局部特征

5. **提取特征**：计算梅尔频谱图（符合人耳感知的频率压缩），MFCC或梅尔滤波器转换为2D特征表示

### 音频分类模型架构

- 输入层：梅尔频谱图(batch,1,freq,time)
- 卷积层，ReLU激活提取局部频谱特征
- 展平层：MaxPool2d池化，将特征图展品为一维向量
- 全连接层，Dropout正则化
- 输出层：输出numclasses各类别概率

### 训练技巧

- Adam优化器收敛快，0.001常用learning_rate初始值
- 设置scheduler动态调整学习率
- EarlyStopping防止过拟合
- 保存最佳模型checkpoint

### troubleshooting

- OOM内存溢出
- 使用预训练模型迁移学习
- 采样率不匹配，统一resample
- 数据增强：time stretch、pitch shift、添加噪音
- 类别不平衡：数据增强，WeightedRandomSampler
- GPU加速，cuda



## Transformer端到端语音识别技术原理

### Whisper

#### 简介

OpenAI推出的ASR模型

#### 整体架构

seq2seq，encoder-decoder架构

- **encoder**
  - 将Log-Mel频谱图作为直接输入
  - 自注意力机制捕捉全局信息，精准捕捉复杂的上下文声学信息
- **decoder**
  - 自回归生成文本token序列
  - masked遮蔽自注意力，生成当前词只能看到已生成的词
  - cross交叉自注意力，将声学特征与文本对齐
- Q来自decoder，代表解码状态；K/V来自encoder，代表声学特征
- 端到端无需对齐组件

### 输入预处理

1. 原始音频分割为30s的片段
2. 转换为对数梅尔频谱图(Log-Mel Spectrogram)，提取关键声学特征，作为encoder输入
3. 将时域信号转为频域图像，更适配卷积网络和transformer

### 多任务训练

- 在训练数据中插入特殊token来指导模型执行不同任务（语音识别、语音翻译、语种辨识）
- 前缀指令序列定义任务目标

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260428150009152.png" alt="image-20260428150009152" style="zoom:50%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260428150207278.png" alt="image-20260428150207278" style="zoom:40%;" />

```py
pip install -U openai-whisper
import whisper
model = ehisper.load_model('base')
result = model.transcribe('audio.mp3')
```



## LangChain构建智能体

### LCEL（LangChain Expression Language）

- 一切都是Runnable协议对象
- 声明式构建
- 原生支持流式streaming、异步和批量处理
- 透明可观测，所有步骤清晰可见

### 环境准备

```PY
pip install langchain-core langchain-openai langgraph
import os
os.environ['OPENAI_API_KEY']='YOUR_API_KEY'
```

### 简单组合

```py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# 定义链
prompt = ChatPromptTemplate.from_template('i请给我讲一个关于{topic}笑话')
model = ChatOpenAI(model='gpt-3.5-turbo')
parser = StrOutputParser()
chain = prompt | model | parser
print(chain)
# 使用invoke执行同步调用
result = chain.invoke({'topic':'programmer'})
print(type(result))
for chunk in chain.stream({'subject': 'starsky'}):
    print(chunk)
```

#### invoke()

- **输入**：接收字典，key值对应prompt模板中的变量名
- **输出**：返回AIMessage对象，包含模型生成的完整回复

#### StrOutputParser

- **解析对象**：模型返回的AIMessage的文本内容
- **核心组件**：用于提取content字段
- **链式拼接**：| 符号

#### 流式输出stream()

- 返回迭代器
- 每次迭代返回一个内容“块”chunk

#### batch()

- 接收一个输入列表，一次性处理多个输入
- LangChain并行执行调用









