# 大作业选题 Projects

以下会议的[截止时间](https://ccfddl.com/)均为 AoE（Anywhere on Earth (UTC-12)，即北京时间第二天的 20:00）。

|  会议名称  |        CCF Rank        |  Abstract  | Submission |
| :--------: | :--------------------: | :--------: | :--------: |
|  AAAI'27   |    CCF-A (人工智能)    | 2026-07-20 | 2026-07-27 |
| BigData'26 | CCF-C (交叉/综合/新兴) |    N/A     | 2026-08-21 |

有意愿投稿的同学欢迎联系课程组和助教，我们会提供相关的辅导和支持。

****

### [CZ-1] KV compression

- **课上教学 KV Cache**
    - [ ] 使用 SnapKV等多种方法，进行KV cache管理
    - [ ] 使用 LongBench进行评测分析，LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding
    - [ ] 基于Mixtral模型，进行管理KV，实验测试效果
- **选题参考 Options**
    - 调研 Surveys：{SnapKV: LLM Knows What You are Looking for Before Generation}


### [ZYX-1] LLM Agents

- **课上教学 Skills taught**
    - [ ] 使用 `openai.OpenAI` client 实现多轮对话 Create a multi-round conversation using `openai.OpenAI`
    - [ ] 使用 `langchain` 实现一个 RAG 应用 Create a RAG-based application with `langchain`

- **选题参考 Options**
    - 调研 Surveys：
        - LLM agents for scientific / engineering / (GPU) kernel tasks
        - OpenClaw / FARS / OpenEvolve etc. workflow & (agent) memory system
        - etc.
    - 开发 Devs：
        - 基于 `langchain` 或 `langgraph` 搭建一个智能体工作流 Implement an agentic workflow using `langchain` or `langgraph`
        - 参考 `optillm` 实现多种单次对话的优化算法 Optimize single round prompting using `optillm`
        - 使用 `openevolve` 求解特定任务 Solve a task using `openevolve`
        - etc.
    - 研究 Researches (in prep. for NeurIPS'26 / EMNLP'26)：
        - 设计新的智能体记忆组件 Design a new agent memory plugin
        - 改进现有的智能体分析方法 Improve existing protocol for agent diagnosis
        - etc.

### [ZYX-2] LLM Post-training

- **课上教学 Skills taught**
    - [ ] 使用 `hf` 传输模型和数据集 Transfer a model or a dataset using `hf` (to Huggingface / ModelScope)

- **预先准备 Prerequisites**
    - 准备好目标任务的数据集（~ 1000 samples） Prepare a dataset with at least ~ 1000 samples

- **选题参考 Options**
    - 使用 LLaMA-Factory 微调模型 Finetuning a model using LLaMA-Factory
    - 使用 OpenRLHF / VeRL 通过强化学习训练模型 Train a model through reinforcement learning (RL) using OpenRLHF / VeRL
    - etc.

### [ZYX-3] Surveys on MLSys / Efficient AI

- **课上教学 Skills taught**
    - [x] (Lesson) Transformers + Attention mechanism

- **选题参考 Options**
    - CUDA Kernel Optimization：FlashAttention, FlashMLA, etc.
    - MoE or Sparse Attention design
    - KV Cache optimization in LLM serving
    - etc.

### [RPZ-1] Autonomous Physics-Informed Neural Networks for Battery State of Health Prediction

- **课上教学 Skills taught**
    - [ ] 使用 `DeepXDE` 或 `Modulus` 构建基础 PINN 模型 Implement a baseline PINN using `DeepXDE` or `Modulus`
    - [ ] 使用 `LangGraph` 实现带有状态机控制的智能体循环 Create an agentic loop with state machine control using `LangGraph`

- **选题参考 Options**
    - 调研 Surveys：
        - LLM Agents 驱动的科学计算 (AI4Science) 综述 LLM Agents for Scientific Computing (AI4Science)
        - 物理信息机器学习中的自动超参数优化与损失平衡策略 Automated HPO and loss balancing in Physics-Informed Machine Learning
        - etc.
    - 开发 Devs：
        - 智能体驱动的特征工程与表征解耦 Agent-driven Feature Mining & Representation Decoupling：
            - 让智能体自动从原始时序数据（如电压、IC 曲线等）中提取最优统计与物理特征，并配置多头注意力机制（Multi-head Attention）以过滤跨域噪声。
        - 自动化物理专家混合网络与门控搜索 Automated Physics-MoE Construction & Gating Search：
            - 参考混合专家网络 (MoE) 架构，构建一个 `Model Architect` 智能体。它能根据 PDE 系统的复杂性，实例化多个独立的“局部物理约束专家”，并编写动态门控单元 (Gating Unit) 实现多专家的自适应加权。
        - 跨域迁移学习自动化与动态损失调优 Automated Transfer Learning & Dynamic Loss Tuning：
            - 实现一个“训练大管家”智能体。它不仅能动态调节 PINN 的数据/物理 Loss 权重，还能在面临新数据集时，自动执行“冻结底层物理约束、解冻表征层”的高效微调 (Fine-tuning) 策略。
        - etc.
    - 研究 Researches：
        - 针对未知电池材料的自动专家扩充 Automated Expert Expansion for Novel Chemistries：研究如何让智能体在面对新电池体系时，自主检索并引入新的物理场控制方程作为新“专家”。
        - 物理场权重演化的可解释性分析 Interpretability of Physical Field Evolution：利用智能体自动追踪和解释电池全生命周期中电化学与机械应力特征的动态演化模式。
        - etc.
