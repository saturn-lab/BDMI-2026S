## 20260414 BDMI课程小记
#### by 物理42 俞善斌
第八次课。

---

### 1. 循环神经网络（RNN）及其变种

#### 基本循环网络（Vanilla RNN）
- 结构：各时间步共享权重矩阵  
  - U：输入层 → 隐藏层  
  - V：隐藏层 → 输出层  
  - W：隐藏层 → 隐藏层（反馈连接）
- 前向计算：
  \[
  h_t = \tanh(Ux_t + Wh_{t-1} + b), \quad o_t = Vh_t + c
  \]
- 训练：通过时间反向传播（BPTT），损失为各时间步损失之和（如负对数似然），内存代价 \(O(T)\)，不可并行。
- 问题：梯度消失/爆炸 → 引入梯度截断。

#### LSTM（长短时记忆网络）
- 增加记忆单元（Cell State）和三个门：
  - 遗忘门 \(f\)：控制是否遗忘
  - 输入门 \(i\)：控制是否写入
  - 输出门 \(o\)：控制是否输出
- 门结构：控制信号经 Sigmoid 后与输入逐元素相乘，门可训练。
- 记忆单元更新：\(C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t\)。
- 优势：解决 RNN 收敛慢问题，提升长距离依赖能力。

#### GRU（门控循环单元）
- LSTM 简化版：合并记忆状态 C 和隐藏状态 H 为单一状态。
- 仅两个门：
  - 更新门 \(z\)：融合输入门和遗忘门，控制历史信息保留
  - 重置门 \(r\)：控制忽略历史信息的程度
- 参数更少，计算更快，所需训练数据少；数据充足时 LSTM 表达能力更强。

#### RNN 结构扩展
- 深度 RNN：堆叠多个循环层。
- 双向 RNN：由正向和反向两个子 RNN 组成，同时依赖过去和未来信息，常用于手写识别、语音识别、NLP，处理延迟大。

#### RNN 应用示例
- 序列建模：字母预测、股价移动预测、天气预测
- 自然语言处理：语音识别、图片注解（Image Captioning）、文本情感分类、文本生成、机器翻译（Seq2Seq）

---

### 2. 计算机视觉核心任务

#### 视觉任务
- 分类（Image Classification）
- 定位（Localization）
- 对象检测（Object Detection）：类别+位置
- 语义分割（Semantic Segmentation）：像素级类别

#### 评价指标
- IOU（交并比）：IOU > 0.5 判为正确
- 精确率（Precision）：\(TP/(TP+FP)\)
- 召回率（Recall）：\(TP/(TP+FN)\)
- mAP：综合精确率与召回率的平均精度

---

### 3. 图像分类方法

#### 卷积网络（CNN）
- 卷积运算：卷积核与输入局部区域的线性加权求和。
- 典型结构：卷积层 + 激活（ReLU/Sigmoid）+ 池化层 + 归一化层 + Dropout。
- 优点：参数少、可并行、计算快。
- 经典网络发展：LeNet → AlexNet → VGG → GoogLeNet → ResNet → DenseNet。

#### Vision Transformer (ViT)
- 将图像分割为固定大小 patch，线性嵌入后加位置编码，送入标准 Transformer 编码器；使用可学习的 [cls] token 进行分类。
- 变体：DeiT、Swin Transformer（分层架构+移位窗口）、LeViT、XciT、MLP-Mixer 等。
- AbSViT：引入自顶向下注意力，可随 prior 变化将注意力集中到相应对象，体现 “综合式分析”（Analysis by Synthesis）。

---

### 4. 对象检测方法

#### R-CNN 系列
- R-CNN：Selective Search 候选区域 + CNN 分类
- Fast R-CNN：RoI Pooling 提升速度
- Faster R-CNN：引入区域生成网络（RPN），使用 anchor 机制，速度与精度大幅提升，训练时有 4 个损失函数。

#### YOLO (You Only Look Once)
- 将检测视为回归问题，网格预测边界框和类别，速度快（Titan X 上 45 FPS）。
- 网络借鉴 GoogLeNet/NIN，输出 \(S \times S \times (B \times 5 + C)\) 张量。
- 历经 YOLOv1 至 YOLOv11 等多个版本。

#### SSD (Single Shot MultiBox Detector)
- 在不同尺度的特征图上用卷积核预测边界框和得分，一次前向完成检测。
- 使用默认框（Default Box）和非极大值抑制（NMS）。

#### DETR
- 基于 Transformer 的端到端检测：CNN backbone 提取特征，经 Transformer 编码器-解码器，用 object queries 直接预测类别和边界框。

---

### 5. 图像语义分割

#### FCN（全卷积网络）
- 将分类网络的全连接层替换为卷积层，通过上采样恢复原图尺寸。
- 融合浅层（细节多）和深层（语义强）特征图，提升分割精度。
- 缺点：未充分考虑像素间联系，细节分割不够好。

#### 后续发展
- SegNet：编解码对称结构 + 上池化
- U-Net：U 型对称结构，特征图拼接，适合大图
- PSPNet：金字塔池化融合多尺度上下文
- DeepLabv3 等

#### 基于 Transformer 的分割
- TransUNet、Swin-UNet、UNETR、HiFormer、TransBTS 等，结合 CNN 局部特征与 Transformer 全局上下文，广泛用于医学图像分割。

---

### 6. 自动驾驶中的视觉应用
- 传感器：摄像头、激光雷达、雷达、北斗/GPS 等，每日约 4TB 数据。
- AI 任务：交通标志检测、行人/车辆检测、车道线识别、可行驶区域感知。
- 英伟达平台：LaneNet（车道线）、PathNet（可行驶路径）、PilotNet（驾驶中心路径）。
- 特斯拉 FSD：纯视觉 HydraNets 多任务网络，8 个摄像头输入融合为 Vector Space。

---

### 7. 注意力机制与 Transformer

#### 注意力机制发明简史
- 2003, Bengio: 用神经网络学习语言模型，预测下一词
- 2013, Graves: 用 LSTM 生成长距离依赖序列
- 2014, Sutskever et al.: Seq2Seq 用 LSTM 编解码，定长向量瓶颈
- 2015, Bahdanau et al.: 引入注意力机制，解码时动态对齐输入
- 2016, Cheng et al.: 内部注意力（intra-attention）
- 2017, Vaswani et al.: Transformer，完全基于注意力机制

#### Transformer 结构
- 编码器：多头自注意力 + 前馈网络，N 层堆叠
- 解码器：掩码自注意力 + 交叉注意力 + 前馈网络，N 层堆叠
- 位置编码：正弦/余弦函数

---

### 8. LLM 推理效率优化

#### KV Cache
- 背景：自回归生成时每步重算历史 K、V，复杂度 \(O(n^2)\)，内存占用大。
- 原理：Prefill 阶段计算并缓存所有层的 K、V；Decode 阶段仅计算新 token 的 Q，复用缓存 K、V，复杂度降至 \(O(n)\)。
- 优化手段：
  - MQA (Multi-query Attention)、MLA (Multi-head Latent Attention, DeepSeek)
  - FlashAttention
  - 自适应缓存管理：H2O、SnapKV、PyramidKV、StreamingLLM 等
  - 推理引擎：vLLM（PagedAttention）、SGLang

#### 硬件加速
- LLM 推理核心操作为 GEMM（通用矩阵乘法）。
- Intel CPU 指令集：
  - AVX-512：512 位 SIMD
  - AMX：专用矩阵乘法加速，Tile 寄存器，支持 BF16/INT8
- 量化：BF16（Bfloat16）、INT8 等降低内存和计算量。

#### 其他推理加速方法
- 稀疏注意力近似（Sparse/Low-rank）：Reformer、Performer、Sparse Transformer、SpAtten
- 缓存策略：LSU/LFU、语义提示缓存（VectorQ 自适应语义缓存）

---

### 9. 多模态视觉大模型

#### ViT
- 图像切块 → 线性嵌入 + 位置编码 → Transformer 编码器，用 [CLS] token 分类。

#### CLIP (OpenAI)
- 双塔：图像编码器（如 ResNet）+ 文本编码器（Transformer）
- 对比学习预训练，将图像和文本映射到同一向量空间，实现零样本分类。

#### MAE（MetaAI）
- 自监督预训练：随机遮掩图像 patch（mask 75%），非对称编解码器（编码器仅处理可见块，解码器重构全图）。
- 高效可扩展，下游分割/检测性能优异。

#### SimCLR
- 对比学习视觉表征框架。

#### 国内大模型
- 提及 CogVLM、GLM4v-9B 等。

---

### 10. Torchvision 对象检测实践
- Torchvision 是 PyTorch 的 CV 扩展库，包含 datasets、transforms、models、ops 四大模块。
- 支持直接加载预训练的 Faster R-CNN、SSD、Mask R-CNN，并进行微调。
- 示例流程：导入库 → 加载模型 → 修改分类头 → 定义 Dataset → 训练和测试。