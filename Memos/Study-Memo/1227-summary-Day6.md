## 20260331 BDMI课程小记
#### by 物理42 俞善斌
第六次课。
- 回顾了 **机器学习基础与scikit-learn实践** 的关键方法  
- 系统介绍了 **PyTorch 深度学习框架** 的核心概念与基础操作  
- 深入讲解了 **多层人工神经网络** 的结构与训练流程  
- 详细阐述了 **卷积神经网络（CNN）** 的基本原理与组件  
- 初步介绍了 **Transformer 架构** 的核心思想与变体分类  
- 梳理了 **模型参数优化** 的完整训练循环  

---

### 1. PyTorch 自动微分与模型构建

#### 自动微分基础  
```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2
y.backward()
print(x.grad)  # 3*x^2 + 4*x = 20
```
- 动态计算图（Define-by-Run）：随代码执行动态构建，灵活直观  
- 对比静态图（Define-and-Run）：先定义后执行，优化更充分  

#### 模型定义示例（逻辑回归）  
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

#### 训练循环三步骤  
```python
for epoch in range(epochs):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### 2. 卷积神经网络（CNN）

#### 基本概念  
- **卷积核（Kernel/Filter）**：可学习的权重矩阵，提取局部特征  
- **填充（Padding）**：边缘补零，控制输出尺寸  
- **步长（Stride）**：卷积核滑动步幅  
- **输出尺寸公式**（无填充）：\(L = \frac{W-F}{S} + 1\)；有填充：\(L = \frac{W-F+2P}{S} + 1\)  

#### 卷积运算示例  
输入张量：3×4，卷积核：2×2，步长=1，无填充 → 输出尺寸 2×3  
```python
# 手动计算示例
Input = [[2,2,2,2],
         [4,4,4,4],
         [8,8,8,8]]
Kernel = [[1,1],
          [2,2]]
# 输出第一行第一列：2*1+2*1+4*2+4*2 = 20
# 输出应为 [[20, 20, 20],
#          [40, 40, 40]]
```

#### CNN 典型结构  
- **卷积层**：提取特征  
- **激活层**：ReLU 引入非线性  
- **池化层**：降采样，减小尺寸（最大池化/平均池化）  
- **批归一化层（BN）**：稳定训练，加速收敛  
- **Dropout 层**：防止过拟合  

#### 优势对比  
- 相比全连接层，CNN 参数少、可并行计算，适合图像等高维数据  
- 例如：200×200×3 图像，全连接单层参数量 12 万，而 CNN 仅由卷积核大小决定  

#### 经典网络演进  
LeNet → AlexNet → VGG → GoogLeNet → ResNet → DenseNet  

---

### 3. Transformer 架构初步

#### 核心机制：自注意力（Self-Attention）  
- 输入序列每个词通过 Query、Key、Value 计算与其他词的关联权重  
- **缩放点积注意力**：  
  \[
  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \]
- **多头注意力**：并行多组注意力，捕捉不同子空间特征  

#### 编码器-解码器结构  
- **编码器**：多头自注意力 + 前馈网络，N 层堆叠  
- **解码器**：掩码自注意力 + 交叉注意力 + 前馈网络  
- **位置编码**：注入序列顺序信息，使用 sin/cos 函数生成  

#### Transformer 三大变体  
| 类型 | 代表模型 | 结构特点 |
|------|----------|----------|
| 仅编码器 | BERT, RoBERTa | 双向上下文理解，适合分类、问答 |
| 仅解码器 | GPT, Llama | 单向自回归生成，擅长文本生成 |
| 编码器-解码器 | T5, BART | 序列到序列任务，如翻译、摘要 |

#### 视觉 Transformer（ViT）  
- 将图像切分为固定大小的 Patch，线性投影后作为序列输入 Transformer 编码器  
- 添加可学习的 `[CLS]` token 用于分类任务  

---

### 4. 模型训练超参数与优化技巧

#### 关键超参数  
- **学习率（Learning Rate）**：控制参数更新步长，过大震荡，过小收敛慢  
- **批次大小（Batch Size）**：每次更新使用的样本数  
- **训练轮数（Epochs）**：遍历整个数据集的次数  

#### 优化器使用流程  
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    for x_batch, y_batch in dataloader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 模型保存与加载  
```python
# 保存
torch.save(model.state_dict(), 'model.pth')
# 加载
model.load_state_dict(torch.load('model.pth'))
```

---