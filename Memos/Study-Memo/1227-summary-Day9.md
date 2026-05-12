## 20260421 BDMI课程小记
#### by 物理42 俞善斌
第九次课。

---

### 1. 深度学习概述

#### 从线性分类器到深度神经网络
- 深度学习部分先回顾了人工智能、机器学习、监督学习、深度学习等术语。
- 图像分类中，线性分类器（Linear Classifier）和 SVM（Support Vector Machine）通过超平面划分类别。
- 训练模型的基本流程：
  - 定义损失函数（Loss Function）
  - 通过优化（Optimization）更新参数
  - 用反向传播（Backpropagation）在计算图上计算梯度
- 线性分类器在复杂图像分类问题上效果有限，因此引入多层人工神经网络。

#### 多层人工神经网络
- 人工神经元来源于生物仿生，常配合激活函数使用。
- 多层全连接网络又称多层感知机（MLP）或 Dense 网络。
- 网络层间连接方式包括：
  - 前馈网络（feedforward）：MLP、CNN
  - 反馈网络（feedback）：RNN
  - 记忆网络（memory network）
- CNN 是局部连接的前馈网络，区别于全连接网络。

#### 深度神经网络训练
- 训练深度网络需要考虑：
  - 数据预处理
  - 网络结构和损失函数设计
  - 梯度下降
  - 学习率选取
  - 过拟合与正则化
- 学习率策略包括指数衰减、\(1/t\) 衰减和分步衰减，也可使用 RMSProp、Adam 等优化器。
- 正则化方法包括 L2 参数正则化、数据增强、Dropout 和 Ensemble Learning。

#### Transformer 回顾
- Transformer 是一种 encoder-decoder 结构，用于并行训练。
- 基本术语：
  - 自注意力（Self Attention）
  - 可缩放点积注意力（Scaled Dot-Product Attention）
  - 位置编码（Positional Encoding）
  - 遮挡（Masking）
  - 多头注意力（Multi-Head Attention）
- 编码器层：多头注意力 + 点式前馈网络，配合残差连接和层归一化。
- 解码器层：遮挡多头注意力 + 多头交叉注意力 + 点式前馈网络。
- Transformer 总结：
  - 与 RNN 不同，可以较好并行训练
  - 本身不能利用词序信息，需要加入位置 Embedding
  - 重点是 Self-Attention 结构
  - Multi-Head Attention 可以捕捉多个维度上的相关系数

---

### 2. RNN 及其扩展

#### 基本循环网络（Vanilla RNN）
- RNN 的特点是循环反馈连接：当前状态会作为输入传递到下一步计算。
- Vanilla RNN 中每个时间步使用同一组权重矩阵：
  - \(U\)：输入层到隐藏层
  - \(V\)：隐藏层到输出层
  - \(W\)：隐藏层到隐藏层
- 训练目标是将输入序列 \(x\) 映射到输出序列 \(o\)，总损失为各时间步损失之和。
- 使用 softmax 输出时，\(o\) 是未归一化的对数分数（logit）。

#### BPTT 与训练问题
- 通过时间反向传播（Back-Propagation Through Time, BPTT）在展开的循环网络计算图上计算梯度并调节权重。
- BPTT 包括一次从左到右的前向传播和一次从右到左的反向传播，运行时间约为 \(O(T)\)。
- 每个时间步顺序计算，前向状态需要保存到反向传播时使用，因此内存代价也是 \(O(T)\)。
- RNN 训练常见问题：
  - 梯度消失（Gradient Vanishing）
  - ReLU 函数问题
  - Sigmoid 函数饱和问题
- 梯度截断（Gradient Clipping）：当梯度过大或过小时，将其限制到固定范围内，例如某维度大于 1 时固定为 1。

#### LSTM 与门结构
- LSTM 是对 RNN 的改进，引入记忆单元（cell / memory cell）和三个门：
  - 输入门（input gate）：控制是否输入
  - 遗忘门（forget gate）：控制是否存储
  - 输出门（output gate）：控制是否输出
- 门结构的输入和控制是形状一致的张量，门本身可以通过反向传播训练。
- LSTM 的核心是记忆单元 \(C_t\)，可以看作 \(C_{t-1}\) 和 \(\tilde{C}_t\) 的线性加权组合。
- LSTM 增加了损失函数导数传播路径，提升了 RNN 处理长距离依赖（Long-term dependencies）的能力。

#### GRU
- GRU（Gated Recurrent Unit）是 Cho 等在 2014 年提出的 LSTM 简化版本。
- GRU 将 LSTM 的记忆状态 \(C\) 和隐藏状态 \(H\) 合并为一个状态。
- GRU 只有两个门：
  - 更新门（update gate）：合并输入门和遗忘门，用于控制历史信息对当前隐层输出的影响
  - 重置门（reset gate）：关闭时忽略历史信息
- GRU 参数更少、计算更快、需要的训练数据较少；数据量足够时 LSTM 可以发挥表达能力强的优点。

#### 深度 RNN、双向 RNN 与应用
- 深度 RNN：堆叠多个循环层形成栈式结构。
- 双向 RNN：由两个 RNN 组成，一个从序列起点向后计算，另一个从序列末尾向前计算。
- 双向 RNN 常用于手写识别、语音识别、自然语言处理。
- RNN 应用：
  - 手写识别
  - 股价移动预测
  - 天气预测
  - 语音识别 ASR
  - 图片注解 Image Captioning
  - 文本情感分类
  - 文本生成
  - 机器翻译

#### 记忆网络、NTM 与 DNC
- 图灵机包括无限纸带、读写头、控制规则和状态寄存器。
- 课件中提到：用 Sigmoid 激活函数的 RNN 是图灵完备的，只要给出正确权重，可以计算任何可计算程序。
- 记忆网络与反馈网络的区别：反馈网络记录前一时刻，记忆网络记录前面所有时刻。
- 神经图灵机（Neural Turing Machine, NTM）组件：
  - 控制器
  - 外部记忆
  - 读写操作
  - 输入输出
  - 整体架构可微分训练
- 可微神经计算机（Differentiable Neural Computer, DNC）是带动态外部记忆的神经网络混合计算结构。

---

### 3. 计算机视觉：目标检测与语义分割

#### 视觉任务与评价指标
- 计算机视觉任务包括分类、定位、检测、分割。
- IOU（Intersection over Union）：预测框与真实框的相交面积 / 相并面积。
- 检测错误类型：
  - Correct：类别正确且 IOU > 0.5
  - Localization：类别正确且 \(0.1 < IOU < 0.5\)
  - Similar：类别相似且 IOU > 0.1
  - Other：类别错误且 IOU > 0.1
  - Background：IOU < 0.1

#### Faster R-CNN
- Faster R-CNN 可以看作 “RPN + Fast R-CNN” 的系统，用 RPN 代替 Fast R-CNN 中的 Selective Search。
- 包含两个模块：
  - RPN（Region Proposal Network）：基于深度卷积层给出矩形候选区域
  - Fast R-CNN RoI 池化层：对 proposal 区域分类并提取定位
- RPN 在特征空间上滑动 \(3 \times 3\) 窗口，判断区域下是否有对象，并给出 bounding box 定位。
- Anchor 候选区域：对特征图每个位置考虑 9 个候选窗口，即三种面积 \(\{128^2, 256^2, 512^2\}\) 与三种比例 \(\{1:1, 1:2, 2:1\}\) 的组合。
- Faster R-CNN 训练包含 4 个损失：
  - RPN 分类是否对象
  - RPN 边界框提议
  - Fast R-CNN 对象分类
  - Fast R-CNN 边界框回归

#### YOLO 与 SSD
- YOLO（You Only Look Once）将目标检测从分类问题化简为回归问题，在保证精度不过多损失的前提下提高检测速度。
- YOLO 基本流程：
  - 将整张图像输入网络，分割为 \(S \times S\) 网格
  - 目标中心落在哪个网格，该网格负责预测类别和位置
  - 每个网格预测 \(B\) 个 bounding box，每个框对应 \((x,y,w,h)\) 和 confidence
- YOLO v1 网络输出为 \(S \times S \times (B \times 5 + \#Classes)\) 的三维张量。
- SSD（Single Shot MultiBox Detector）基于前向传播 CNN，产生固定大小的 bounding boxes 和每个框中包含对象的分数。
- SSD 在不同层次的 feature maps 上预测 object、box offsets，并使用 Non-maximum suppression 得到最终预测。

#### DETR
- DETR（DEtection TRansformer）使用 Transformer 做端到端目标检测。
- CNN backbone 提取图像特征后，加入空间位置编码并输入 Transformer Encoder。
- Decoder 接收 object queries、输出位置编码和 encoder memory，最终预测类别标签和边界框。

#### 语义分割
- CNN 图像分类：卷积和池化后将特征图转为向量，再送入 softmax。
- FCN 语义分割：通过上采样将特征图还原到原图大小，再输出每个像素的概率分布。
- FCN 的优点：把图像级分类扩展到像素级分类，具有里程碑意义。
- FCN 的缺点：对像素独立分类，没有充分考虑像素之间联系，细节分割不够好。
- 后续模型：
  - SegNet：Encoder-Decoder 对称结构和上池化层
  - PSPNet：融合不同尺度池化特征，分析上下文信息
  - DeepLabv3
  - keras2_segmentation：包含 FCN、SegNet、Unet、PSPNet 等实现
- Transformer based segmentation：
  - TransUnet：将 Transformer 集成到基于 CNN 的模型中
  - Swin Transformer：分层架构和 shifted windows
  - Swin-UNet：纯 Transformer 架构，没有 CNN 模块
  - UNETR：编码器中使用 Transformer 块，解码器中使用 CNN，用于 3D 医学图像分割
  - HiFormer：融合 CNN 和 Transformer 特征

---

### 4. 自动驾驶中的视觉应用

#### 自动驾驶与 ADAS
- 自主驾驶（Self-Driving / Driverless）：通过控制车辆速度、方向和刹车接替人类驾驶员。
- 辅助驾驶（ADAS）：利用激光雷达、相机、GPS 等传感器观察环境，再由决策算法提醒驾驶员。
- 自动驾驶传感器包括：
  - 激光雷达
  - 摄像头
  - 声纳雷达
  - 北斗 / GPS 定位
- 课件中提到传感器数据量约为 4TB / 天。

#### 自动驾驶 AI 任务
- 交通标志与信号灯检测
- 路人 / 行人检测
- 车辆检测
- 车道线和转弯线识别
- 驾驶员人脸识别

#### NVIDIA 与 Tesla 示例
- NVIDIA 道路感知：
  - LaneNet：预测车道线
  - PathNet：预测可运行路径边线，无论是否有车道线
  - PilotNet：根据人类驾驶轨迹预测驾驶中心路径
- Tesla FSD：
  - 纯视觉算法 HydraNets 使用 8 个摄像头输入
  - 输入规格为 \(1280 \times 960\) 12-Bit HDR 36Hz
  - 单个神经网络整合为 3D 环境感知，称为 Vector Space
  - 基于深度卷积网络识别视觉内容，具有多任务学习能力

---

### 5. 多模态视觉大模型

#### ViT
- ViT（Vision Transformer）将图像切成 patch，以便使用原始 Transformer 结构。
- 在 encoder 第一个位置加入 cls-token，用第一个位置 encoder 输出的特征进行分类器训练。
- 具体流程：
  - 切分图像并将 patch 拉平到一维
  - 通过线性映射得到 patch embeddings
  - 加入位置 embedding
  - 输入 Transformer Encoder

#### Google SMoE 与 SimCLR
- Google 视觉大模型部分提到 ViT-G/14 和 “Scaling Vision with Sparse Mixture of Experts”。
- SimCLR 是 “A Simple Framework for Contrastive Learning of Visual Representations”，即视觉表示对比学习的简单框架。

#### CLIP
- CLIP（Contrastive Language-Image Pre-training）属于多模态学习，采用文本作为监督信号。
- 核心思想：将文本和图像映射到共同隐空间，实现语义对齐。
- 模型结构包括：
  - 图像编码器（Image Encoder）：将图像映射为高维嵌入表示
  - 文本编码器（Text Encoder）：将文本描述映射到同一向量空间
- CLIP 训练流程：
  - Contrastive pre-training：使用图片-文本对进行对比学习
  - Create dataset classifier from label text：提取预测类别文本特征
  - Use for zero-shot prediction：进行零样本推理预测

#### MAE
- MAE（Masked Autoencoders）把 BERT 中遮掩语言模型的思想引入 CV，通过遮掩图像 patch 并重构缺失像素进行自监督训练。
- 两个核心设计：
  - 非对称编码器-解码器架构
  - 高遮掩率（mask 75%）
- 非对称设计：编码器只处理可见 patch，被遮掩块不编码；解码器重构所有块。
- 解码器更轻量，课件中提到其计算量仅为编码器的 10%。
- 损失函数：用 MLP 将重构特征映射回像素值，计算与真实像素值的 MSE。
- MAE 在目标检测、实例分割、语义分割等任务上表现优异。

#### 国内视觉语言模型
- 课件提到 CogVLM 和 GLM-4V-9B。
- GLM-4V-9B 是智谱 AI 于 2024 年开源的 90 亿参数视觉语言模型。
- ModelScope（魔搭社区）被介绍为面向中文 AI 开发者和应用场景的一站式模型服务平台。

---

### 6. LLM API 调用

#### 基本流程
- 使用 LLM API 的步骤：
  - 到云计算平台申请 API_KEY
  - 确认 BASE_URL
  - 根据模型 ID 调用大模型
  - 解析大模型输出并驱动工具调用
  - 将工具结果整合进上下文再次调用大模型
  - 优化工具调用与大模型调用的工作流
- 课件示例平台包括清华·并行智算云和华为云 ModelArts。
- 学习笔记只记录流程，不记录课件中的具体密钥。

#### 消息角色与多轮对话
- System：最高优先级的控制层指令。
- User：用户请求，即 task specification。
- Assistant：模型过去的响应，即 stateful trace。
- 多轮对话中，ChatState 会把 system message 和历史消息打包成 messages 传给大模型。

#### 上下文长度与调用方式
- 超过模型上下文长度时，早期记录会被舍弃（First-In-First-Out），可能导致后续任务不准确。
- 解决方法：
  - 将上一步输出整合进当前任务输入：Context Engineering
  - 压缩上下文，只保留重要部分：Agentic Memory
  - 推理时仅保留与任务最相关 token：Sparse Attention
- 同步调用：等待当前请求处理完再发送新请求。
- 异步调用：同时提交请求，哪个先完成先返回。

#### 接口结构
- ChatState：大模型对话历史记录。
- ChatTransport：大模型对话调用。
- ChatSession：大模型对话接口。
- 用户通过 `ask` / `ask_async` 发送提示词，ChatTransport 负责请求大模型并整合输出。

---

### 7. Mamba 与状态空间模型

#### 序列建模目标
- 课件比较了 RNN、CNN、Transformer 三类序列模型。
- 希望模型同时具备：
  - 像 Transformer 一样可并行训练
  - 像 RNN 一样对长序列具有 \(O(N)\) 计算和内存成本
  - 推理每个 token 时具有 \(O(1)\) 的常数计算和内存成本

#### State Space Model
- 状态空间模型（SSM）通过状态表示 \(h(t)\) 将输入信号 \(x(t)\) 映射到输出信号 \(y(t)\)：
  \[
  h'(t) = Ah(t) + Bx(t)
  \]
  \[
  y(t) = Ch(t) + Dx(t)
  \]
- 该形式是线性且时不变的，因为参数矩阵 \(A,B,C,D\) 不随时间变化。
- 连续信号需要离散化才能处理离散序列，课件中提到论文使用 ZOH（Zero-Order Hold）规则离散化系统。
- 离散化步长 \(\Delta\) 在实践中可以作为模型参数，用梯度下降学习。

#### 递归计算与卷积计算
- 递归形式适合推理：每次生成一个 token，每个 token 使用常数计算和内存。
- 卷积形式适合训练：已经拥有完整输入序列时，可以并行计算。
- SSM 的输出可以看作 kernel \(K\) 与输入 \(x(t)\) 的卷积。
- 多维输入时，每个维度可以由独立的 SSM 处理，类似 Transformer 中不同 head 管理不同维度组。

#### HiPPO 与 Mamba 动机
- \(A\) 矩阵用于从前一状态捕获信息并构建新状态，决定历史信息如何向前复制。
- HiPPO 理论用 Legendre polynomials 近似到目前为止的输入信号，对近期信号表示更精确，旧信号指数衰减。
- Vanilla SSM / S4 的不足在于时间不变，参数 \(A,B,C,D\) 对每个 token 相同，难以进行 content-aware reasoning。
- Mamba 是 selective SSM，模型参数随输入 token 变化。

#### Selective Scan 与实现优化
- Mamba 因为参数随 token 变化，不能直接用卷积方式求值，否则需要为长度 \(L\) 构造 \(L\) 个不同卷积核。
- Scan operation 类似 prefix-sum，每个状态由前一状态和当前输入计算。
- 若操作满足结合律，scan 可以通过 parallel scan 并行化。
- Mamba 的 Selective Scan 使用递归形式，并借助 parallel scan 加速。
- 课件还提到：
  - Kernel Fusion：将多个 CUDA kernel 融合，减少 HBM 与 SRAM 之间的数据搬运
  - Recomputation：反向传播时重新计算激活，减少缓存和读写开销
  - Mamba block 可堆叠成类似 Transformer 层堆叠的模型结构

---

### 8. Torchvision 对象检测实践

- Torchvision 是 PyTorch 官方面向计算机视觉任务的扩展代码库。
- 主要功能：包含经过训练的视觉模型。
- 模块功能包括：
  - 经典视觉数据集下载与加载，如 ImageNet、CIFAR-10、COCO
  - 图像变换与预处理
  - 预训练视觉模型
  - 面向 CV 的高效计算操作，如 NMS、RoIAlign
- 可以使用 Faster R-CNN 或 SSD 等模型检测图像中的对象。

---

### 9. 时频谱图补充阅读：语音情感识别

#### ViT 用于语音情感识别
- 补充阅读提出使用轻量级 Vision Transformer 改进 Speech Emotion Recognition（SER）。
- 方法：将语音转为 mel spectrogram，把时频谱图作为图像输入 ViT。
- ViT 通过自注意力捕捉空间依赖和高层特征，用于识别语音中的情感状态。
- 实验数据集包括 TESS 和 EMODB。
- 结果中给出准确率：
  - TESS：98%
  - EMODB：91%
  - TESS-EMODB：93%
- 消融实验提到：去掉 dropout layer 会降低准确率；patch size 变化会影响总体准确率和计算复杂度，32 patch size 是文中采用的较优选择。

#### ESERNet
- ESERNet 面向 classroom discourse analysis 中的语音情感识别。
- SER 在课堂场景中可帮助教师了解学生情绪并调整教学活动。
- 主要挑战：
  - 情绪本身具有模糊性
  - 噪声环境下从语音中解释情绪更复杂
- ESERNet 是基于 Transformer 的双路径框架：
  - crucial cue pathway：通过恢复 masked spectrogram 学习关键特征
  - relationship pathway：通过 Swin Transformer 获取多尺度特征并学习长距离依赖
- 数据集包括 IEMOCAP 和 EmoDB。
- 结论中强调：关键情感线索和不同情绪之间的微小差异，是提升 SER 性能的重要因素。
