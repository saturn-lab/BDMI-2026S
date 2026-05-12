## LSTM长短时记忆网络（long short-term memory network）

为解决RNN的长距离信息丢失的问题

### 门控结构（Gate structure）

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421134617707.png" alt="image-20260421134617707" style="zoom:26%;" />

输入和控制是形状一致的张量，控制经过sigmoid函数后，变为0~1之间的张量

### 记忆单元和门输入单元

LSTM 每个时间步有两个状态：

- **隐藏状态** $h_t$：对外输出的当前信息
- **细胞状态** $C_t$：沿时间主线传递的长期记忆

1. memory cell：记得比hidden state更远，寄存时间序列的输入
2. **遗忘门（output gate）**：决定丢掉哪些旧记忆
3. **输入门（input gate）**：决定写入哪些新信息
4. **输出门（output gate）**：决定当前输出什么

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421134831466.png" alt="image-20260421134831466" style="zoom:33%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421134959024.png" alt="image-20260421134959024" style="zoom:33%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421135410261.png" alt="image-20260421135410261" style="zoom:33%;" />

### concatenate连接

合并为一个矩阵$W_{xh}x_t+W_{hh}h_{t-1}\rightarrow W_i \cdot [h_{t-1},x_t]$

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421135657419.png" alt="image-20260421135657419" style="zoom:33%;" />

![image-20260421135948806](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421135948806.png)

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421140021709.png" alt="image-20260421140021709" style="zoom:33%;" />

### 核心原理

- 记忆单元（Cell）是LSTM的核心，作用是对时间的一个累加（积分器）
- Ct看成是Ct-1和C_t的线性加权组合
- 为了防止Ct爆炸，C_t是通过sigmoid函数或者tanh函数
- 通过Ct计算出来ht
- 损失函数的导数传播增加了一条路径
- 损失函数的导数如果不能通过C_t进行传播，可以通过C_t-1传播

## GRU(gated recurrent unit)

LSTM的简化版本，最早用于机器翻译和语音识别

### 架构

合并输入门和遗忘门为更新门（update gate），更新门越大，更多保留旧信息，少接收新信息

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421140815097.png" alt="image-20260421140815097" style="zoom:33%;" />

reset gate重置门，关闭即忽略历史信息

### 优势

参数量更小，计算更快，性能相差不大

### 应用

1. 语音识别ASR
2. 图片注解Image Captioning
3. 文本情感分类text sentiment classification
4. 文本生成text generation

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421141807549.png" alt="image-20260421141807549" style="zoom:33%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421141818703.png" alt="image-20260421141818703" style="zoom:33%;" />

## 计算机视觉

### IOU重叠联合比

预测框和真实框的重叠联合比

$IOU = \dfrac{A_{intersection}}{A_{union}}$

- 正确（Correct）：类别正确，IOU > 0.5
- 定位错误（Localization）：类别正确，0.1 < IOU < 0.5
- 相似性错误（Similar）：类别相似，IOU > 0.1
- 其他错误（Other）：类别错误，IOU > 0.1
- 背景误认（Background）：IOU < 0.1

### 目标检测

#### R-CNN(region-based convolutional network method)

提取region proposals

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421145047615.png" alt="image-20260421145047615" style="zoom:33%;" />

深度卷积层基础上给出一系列的矩形候选区域

用最后一个卷积层推断候选区域

#### RPN（region proposal network）

- 检测图像中有没有对象，提出方框区域
- 在特征空间上滑动一个窗口（anchors box）

#### YOLO（you only look once）

- 由分类问题化简为回归问题
- 运算速度高

- 输入的图像分割为S*S个grid，每个网格负责预测边界框的5个参数（中心点坐标、宽度高度、置信度）
- 置信度综合反应边界框中存在目标的可能性以及目标位置预测的准确性（非极大值抑制）

#### SSD对象检测算法

- 预测对象及其归属类别的score，在特征图用小卷积核预测一系列的bounding box

#### DETR

- 用transformer的端到端目标检测

### 图像语义分割（semantic segmentation）

将图像中的视觉信息分成不同类别

#### FCN（*）

- 将分类网络的最后全连接层改成卷积层
- 多重卷积与池化层之后，将特征图上采样（卷积核大于原图）还原回原来的大小，再softmax输出每个像素的概率分布
- skip connection对浅层特征图和深层特征图（浓缩程度较高）进行融合相加，弥补下采样中产生的信息损失

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421150821985.png" alt="image-20260421150821985" style="zoom:33%;" />

#### SegNet

- encoder-decoder对称结构，上池化层
- 减小了模型参数，提升了边缘刻画能力

#### UNet

- 特征图融合从相加变为相叠

- 多尺度特征图融合

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421151513231.png" alt="image-20260421151513231" style="zoom:33%;" />

#### PSPNet

- 融合了不同尺度的池化特征
- 更好分析图像上下文信息

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421151637354.png" alt="image-20260421151637354" style="zoom:33%;" />

#### Swin UNet

引入分层架构和移位窗口的机制

#### UNETR

利用编码器结构的Transformer块和解码器的CNN来分割医学图像

## Torchvision

PyTorch官方专为计算机视觉打造的扩展代码库

### 核心模块

- torchvision.datasets (数据集)
  - 内置了大量经典视觉数据集的下载与加载接口（如 ImageNet, CIFAR-10, COCO 等），方便快速验证算法。
- torchvision.transforms (数据处理与增强)
  - 提供了一系列图像预处理工具（如随机裁剪、旋转、颜色变换、张量转化等），用于扩充数据样本，提升模型泛化能力。

- torchvision.models (模型库)
  - 包含众多经典的、预先搭建好的网络架构代码及预训练权重。

- torchvision.ops (特定算子)
  - 提供专门针对计算机视觉任务优化的高效计算操作（如非极大值抑制 NMS、RoIAlign）

### 对象检测

- fasterRCNN或SSD

- box and mask

## 视觉大模型

### ViT

- 将图像切分为小的Patch来更好地使用原始的Transformer结构
- 第一个位置encoder出来的特征作为分类用的特征

### SimCLR

视觉表示对比学习的简单框架

### CLIP

采用文本作为监督信号，将文本和图像映射到一个共同的隐空间，实现语义对齐

zero-shot推理预测

### MAE 是什么

MAE（Masked Autoencoder，遮掩自编码器）是一种面向视觉任务的自监督学习方法，核心思想是将 **BERT 的 Masked Modeling** 引入图像领域： 
把输入图像切分为若干 **patch**，随机遮住其中大部分，只让模型根据剩余可见部分去重建被遮住的内容

---

#### 核心架构

MAE 的整体结构是一个 **非对称的 Encoder-Decoder**：

#### Encoder（编码器）

- 只接收 **未被遮挡的 patch**
- 先将图像切分成 patch
- 再经过 **Patch Embedding**
- 加上 **Position Embedding**
- 输入 **Transformer Encoder** 得到潜在特征表示

#### Decoder（解码器）

- 接收编码器输出的可见 patch 特征
- 再补上 mask 区域对应的 **mask token**
- 结合位置编码后输入轻量级 **Transformer Decoder**
- 最后通过线性层或 MLP 重建所有 patch 的像素值

---

###  MAE 的关键设计

#### 高遮掩率

- MAE 通常采用 **75% 左右的 mask ratio**
- 图像本身冗余较高，因此高比例遮挡仍然可学习
- 高遮挡率可以迫使模型学习更强的全局语义表示，而不是只记局部纹理

#### 非对称结构

- **编码器只处理可见 patch**
- **解码器负责重建全部 patch**
- 这样能显著减少编码阶段的计算量，提高预训练效率

#### 轻量解码器

- 解码器只在预训练时使用
- 下游任务通常只保留编码器
- 因此解码器可以设计得更轻，避免额外增加太多计算开销

---

### 训练流程

#### 预训练阶段

1. 输入图像并切分为 patch
2. 随机遮住大部分 patch
3. 编码器处理剩余可见 patch
4. 解码器接收编码结果和 mask token
5. 重建原始图像 patch
6. 计算重建损失，通常使用 **MSE** 作为目标函数

#### 微调阶段

- 去掉解码器
- 保留预训练好的编码器
- 在分类、检测、分割等下游任务上接任务头进行微调

---

###  MAE 的优势

#### 训练高效

由于编码器只处理少量可见 patch，计算成本明显降低。

#### 可扩展性强

MAE 适合扩展到更大的 ViT 模型和更大规模的数据集。

#### 自监督能力强

不依赖人工标签，仅通过重建任务就能学到有效视觉表征。

#### 下游迁移效果好

在图像分类、目标检测、语义分割、实例分割等任务中都有很强表现。

---

### 与传统 AutoEncoder 的区别

传统自编码器通常对 **完整输入** 编码再重建；  
MAE 则是：

- 只编码 **可见部分**
- 用少量可见信息推断大量缺失内容
- 更强调高层语义理解而不是简单复制输入

因此，MAE 更适合作为视觉大模型的预训练方法

## 大模型调用（LLM API）

### 流程

- 密钥API_KEY

- 平台调用API的网址BASE_URL
- 实现接口、根据模型ID调用
- 解析大模型的输出、驱动工具调用
- 将工具结果整合进上下文再次调用大模型
- 优化工具调用与大模型调用的工作流

```py
import openai

API_KEY = "sk-KUP5iUXThgpibU5fznCsSYRj5n3DHH"
BASE_URL = "http://166.111.238.55:11800/v1"

client = openai. OpenAI(api_key=API_KEY, base_url=BASE_URL)
```

### 消息角色

#### 1. System

- 定义行为边界、任务策略、风格语气等
- 优先级高于 user，且在整个对话过程中持续生效

---

#### 2. User

- 提供问题、指令或数据，让模型执行任务
- 是模型直接响应的主要输入
- 通常承担具体任务说明（task specification）的角色

---

#### 3. Assistant

- 表示对话历史，即模型过去已经产生的响应记录
- 常用于多轮对话中的上下文延续

### 灾难遗忘

- 超过模型上下文长度
- 早期的记录被舍弃(FIFO)
- **解决方法**：
  - context engineering：上一步输出整合进当前任务输入
  - agentic memory：压缩上下文，只保留重要部分
  - sparse attention：推理时仅保留与任务最相关的token

### 异步/同步调用

- 同步：当前请求处理完再发送新的请求
- 异步：同时提交请求，哪个先完成先返回
- <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260421164025386.png" alt="image-20260421164025386" style="zoom:40%;" />
- 多开几个client会增加轮询产生的本地资源开销







