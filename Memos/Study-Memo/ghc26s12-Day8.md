## LaTex

overleaf在线的LaTex编辑器
\[
f(x)=
\begin{cases}
x^2, & x \ge 0 \\
-x, & x < 0
\end{cases}
\]

\[
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\]

- `pmatrix`：圆括号矩阵
- `bmatrix`：方括号矩阵
- `Bmatrix`：大括号矩阵
- `vmatrix`：单竖线
- `Vmatrix`：双竖线

| 模型类型        | 编码器 | 解码器 | 自注意力                      | 交叉注意力 | 输入     | 输出            | 典型任务                 | 代表模型      |
| --------------- | ------ | ------ | ----------------------------- | ---------- | -------- | --------------- | ------------------------ | ------------- |
| Encoder-Decoder | ✅      | ✅      | 编码器：✅<br>解码器：Masked ✅ | ✅          | 源序列   | 目标序列        | 机器翻译、摘要、对话生成 | T5, BART      |
| Encoder-only    | ✅      | ❌      | ✅                             | ❌          | 完整序列 | 每个 token 表示 | 文本分类、句子匹配、检索 | BERT, RoBERTa |
| Decoder-only    | ❌      | ✅      | Masked ✅                      | ❌          | 前缀序列 | 下一个 token    | 语言建模、文本生成       | GPT 系列      |

## Attetnion机制

### 历史

LM language model的发展

![image-20260414140326122](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260414140326122.png)

- 统计语言模型（N-gram）：根据前n-1个词预测下一个单词或字符
- 神经网络NN学习语言模型——用前面连续单词预测下一个词出现的概率（词表示为连续向量）

- LSTM（长短期记忆循环神经网络）生成具有长距离结构的的复杂序列
- 机器翻译NMT，encoder-decoder架构
  - 编码器将输入转换为中间向量
  - 解码器将中间向量转换为翻译的结果
  - 所有信息压缩进中间向量，细节丢失
- Seq2Seq在序列变长时性能下降
- 注意力机制Attention
  - 每一步翻译都关注输入序列的不同部分，捕捉更长距离的依赖（更相关的位置）
  - 动态对齐
  - 端到端
  - 内部注意力（intra-attention）
  - 自然语言推理（NLI）确定前提和假设至今啊的蕴含和矛盾（entailment and contradiction）
  - 计算复杂度高$O(n^2)$，sparse attention等减小计算



## KV Cache（键值缓存）

### LLM推理面临的巨大挑战

1. **内存墙（memory wall）**：模型参数和KV Cache占用大量内存
2. **延迟敏感（Latency Sensitive）**：自回归特性，并行度低
3. **带宽瓶颈（Bandwidth Bottleneck）**：参数和KV Cache需要在HBM和计算核心间频繁移动，成为性能瓶颈
4. **二次方复杂度**：注意力机制的计算成本随输入序列长度的二次方增长，限制了长文本推理能力

###  目的

- 避免重复计算（生成第 i+1个token时前 i 个token的Key和Value已经计算过且保持不变
- 改进推断效率$O(n^2)\rightarrow O(n)$

### 定义

- 保存每一层计算过的注意力键和值的张量，称为KV缓存

### 原理

- 用显存换计算效率
- 历史token的KV保存，只计算新token的K V

### 步骤

- **Prefill Stage**：把已有提示词Prompt整段送入模型，并行计算所有token的QKV矩阵，建立初始 KV Cache

- **Decode Stage**：之后逐 token 生成，每一步复用已有缓存

### 瓶颈

1. 巨大的内存占用
2. 内存带宽限制造成HBM和SRAM间频繁数据搬运导致的巨大访问开销

### 优化KV Cache方法

#### FlashAttention

- 通过分块(Tiling)计算，改变注意力计算算法，减少对HBM的读写频率，KV块加载到SRAM，流式处理Q块
- fused kernel核函数融合，将attention计算融合为一个CUDA Kernel，中间结果在SRAM流转

#### MQA/GQA

- **Multi-Query Attention** 所有Query头共享一组KV头
- **Grouped-Query Attention** 多组Query头共享一组KV头
- 减小KV Cache体积和内存带宽需求<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260428161743450.png" alt="image-20260428161743450" style="zoom:33%;" />

#### Sparse / Sliding Window Attention

- 假设最近的token最重要
- 让 query 只关注前面一个固定大小窗口的token，而不是全部历史 token，以减少计算和缓存压力

#### StreamingLLM

- 混合缓存，保留最初几个token的KV Cache，再加上一个最近token的滑动窗口

#### H2O（Heavy Hitter Oracle）

- 基于注意力得分，平衡最近token和历史上重要token的留存

#### PyramidKV

为底层（关注局部语法）分配更多缓存，为高层（关注语义抽象）分配较少缓存

#### SnapKV

基于注意力分数的动态压缩

#### MLA

DeepSeek 的 **Multi-head Latent Attention**，核心是通过低秩联合压缩 K/V，缓解推理时 KV Cache 的瓶颈



### 推理引擎

- **vLLM**：强调高性能内存管理、高并发服务能力
- **SGLang**：强调推理速度和可控性

## 硬件优化推理效率

### 本质

优化LLM推理效率本质是在优化**矩阵乘法（GEMM, General Matrix Multiply）**

### 数值表示

不同精度的数据格式会直接影响推理速度、显存占用和硬件适配方式

### 优化方式

#### AVX（advanced vector extensions）

CPU上广泛使用的通用向量加速手段

#### AMX（advanced matrix extensions）

- 专门面向矩阵乘法加速
- 提供二维的tiled registers
- 比AVX更接近GEMM





## RNN

### 定义

循环的反馈连接，当前的状态作为输入传递到下一步的计算

### 特点

权重共享（每一个时间步处理时的矩阵权重(U、V、W相同）

### 架构

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260414151521234.png" alt="image-20260414151521234" style="zoom:33%;" />

$h(t) = tanh(b+Wh(t-1)+Ux(t))$

$y(t) = softmax(c+Vh(t))$

#### 变式

1. Deep RNN

   <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260414152351081.png" alt="image-20260414152351081" style="zoom:35%;" />

2. 双向RNN：

   - 时间总序列七点开始移动和序列末尾开始移动两个RNN
   - 用于语言识别
   - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260414152555886.png" alt="image-20260414152555886" style="zoom:33%;" align='left'/>
   - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260414152655775.png" alt="image-20260414152655775" style="zoom:33%;" /><img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260414152640935.png" alt="image-20260414152640935" style="zoom:33%;" />

#### LSTM（长短时记忆网络）

解决RNN训练收敛慢的问题

#### GRU（门控循环单元网络）

解决LSTM计算成本高的问题

### 缺点

- 串行计算，后一个隐藏状态依赖前面所有的隐藏状态，无法并行
- **梯度消失**（gradient vanishing）
  - 梯度截断（gradient clipping）：比如<0的数值设定为0；>1的数值设定为1





## 计算机视觉

### 任务流程

摄取图像——数字化信号——分类、定位、检测、分割

### 视觉识别指标

- 精确率（precision）/灵敏度（sensitivity）：$\dfrac{TP}{TP+FP}$假阳性
- 召回率（recall）：$\dfrac{TP}{TP+FN}$假阴性
- 特效度（specificity）：$\dfrac{TN}{TN+FP}$排除错误样本
- 准确率（accuracy）：$acc = \dfrac{TP+TN}{All}$

- AUC（PR（precision-recall）曲线面积大小）

### openCV

```python
pip install opencv-python
import cv2
```

### 传统方法

- 图像分割：分水岭算法，边缘转化为山脉，均匀区转化为山谷，进行目标分割；三角剖分算法；关键点跟踪
- 局部特征点检测（斑点blob；角点corn）
- 基于传统机器学习方法，SVM、Bayes、Dtrees、Boosting
- Haar分类器用于人脸识别

### 深度学习

- ImageNet图像分类赛，ILSVRC图像分类挑战赛
- LetNet-5手写字体识别
- AlexNet：层数更深，参数更多，dropout
- NiN 引入1*1卷积层（Bottleneck layer）和全局池化；
- VGG将7\*7卷积核替换成3个3\*3卷积核，起到了降参数的作用；
- GoogLeNet引入了Inception模块；
- **ResNet**引入了残差思想，增加了Skip Connection；
- DenseNet的DenseBlock中将当前层的输出特征，与之后所有的层做直连

### Transformer

- ViT：远程依赖性，全局上下文能力
- DeiT响应对更资源高效的需求（蒸馏大模型）
- Swin Transformer：引入分层架构和移位窗口
- LeViT：输入图像分割为多个级别或图块
- XciT：沿通道维度而不是令牌维度，交叉协方差注意力机制
- MLP-Mixer：两个全连接层，激活函数用GELU

- Analysis by Synthesis（AbS）
  - **bottom-up**：低层特征到高层语义
  - **top-down**：任务导向、先验，自然语言决定attention区域
- AbSViT：想要识别哪个物体，就会识别哪个物体；贝叶斯推断 其实是在每一层计算一个受top-down signal调控的视觉注意力