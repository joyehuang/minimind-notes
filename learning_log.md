---
title: 学习日志
description: 记录 MiniMind 学习历程中的每日进度、问题和思考
---

# MiniMind 学习日志

> 记录每日学习进度、遇到的问题和解决方案

---

## 📅 学习日志

### 2025-11-06：环境搭建 + 首次运行模型 + 性能测试

#### ✅ 完成事项
- [x] 克隆项目代码
- [x] 解决依赖安装问题（Python 3.13 兼容性）
- [x] 创建虚拟环境并安装所有依赖
- [x] 创建 CLAUDE.md 文档（AI 助手指南）
- [x] 下载预训练模型（MiniMind2，104M 参数）
- [x] 解决 Git LFS 大文件下载问题
- [x] 成功运行模型并完成第一次对话测试
- [x] 进行 CPU vs MPS 性能测试
- [x] 理解模型推理机制和速度影响因素
- [x] 创建学习笔记系统（notes.md）

#### 🐛 问题与解决方案

**问题 1：依赖安装失败 - scikit-learn 编译错误**

**错误信息**：
```
error: subprocess-exited-with-error
× Preparing metadata (pyproject.toml) did not run successfully.
ERROR: Command `/usr/bin/env python ...version.py` failed with status 127
```

**根本原因**：
- Python 3.13 与 scikit-learn 1.5.1 的构建工具不兼容
- requirements.txt 使用严格版本锁定（`==`）

**解决步骤**：
1. 创建修复版 requirements：
   ```bash
   # 将所有 == 改为 >=
   # 例如：scikit_learn==1.5.1 → scikit-learn>=1.5.1
   ```

2. 重新安装：
   ```bash
   pip install -r requirements_fixed.txt
   ```

3. 结果：成功安装兼容版本（scikit-learn 1.6.1）

**经验教训**：
- 对于教育项目，使用 `>=` 版本约束更灵活
- 新版 Python 可能需要更新的包版本
- 预编译的 wheel 包比从源码编译更可靠

---

**问题 2：ModuleNotFoundError: No module named 'torch'**

**错误信息**：
```
ModuleNotFoundError: No module named 'torch'
```

**根本原因**：
- 系统中有多个 Python 安装
  - `python3` 指向：`/opt/homebrew/bin/python3` (Homebrew)
  - `pip` 指向：`/Library/Frameworks/Python.framework/.../pip` (官方安装)
- 用 `pip` 安装的包在一个环境，但 `python3` 运行时在另一个环境

**解决步骤**：
1. 创建虚拟环境：
   ```bash
   python3 -m venv venv
   ```

2. 激活并安装依赖：
   ```bash
   source venv/bin/activate
   pip install -r requirements_fixed.txt
   ```

3. 使用虚拟环境运行：
   ```bash
   source venv/bin/activate
   python eval_llm.py --load_from ./MiniMind2
   ```

**经验教训**：
- **永远使用虚拟环境**：避免环境冲突
- macOS 系统特别容易出现多 Python 环境问题
- Homebrew Python 有 PEP 668 保护，不允许全局安装包

---

**问题 3：SafetensorError: header too large (模型文件未真正下载)**

**错误信息**：
```
safetensors_rust.SafetensorError: Error while deserializing header: header too large
```

**根本原因**：
- `git clone` 只下载了 Git LFS 指针文件（134 字节）
- 真实模型文件（217MB）需要额外下载
- 查看文件发现只有 3 行文本：
  ```
  version https://git-lfs.github.com/spec/v1
  oid sha256:...
  size 217908728
  ```

**解决步骤**：
1. 安装 git-lfs：
   ```bash
   brew install git-lfs
   git lfs install
   ```

2. 拉取真实文件：
   ```bash
   cd MiniMind2
   git lfs pull
   ```

3. 验证文件大小：
   ```bash
   ls -lh model.safetensors
   # 应该显示 208M，不是 134B
   ```

**经验教训**：
- HuggingFace 模型仓库使用 Git LFS 管理大文件
- 下载后记得检查文件大小是否正确
- 可以用 `file` 命令检查文件类型（二进制 vs ASCII 文本）

---

#### 💭 个人思考

**2025-11-06：首次运行感受**
- **惊喜**：一个只有 104M 参数的模型居然能生成 Python 代码！
- **思考**：和 ChatGPT 相比，回答虽然简单，但对于学习 LLM 原理已经足够了
- **疑问**：这么小的模型是如何"记住"这么多知识的？→ 下一步想了解 Transformer 架构
- **性能测试**：
  - 硬件：Apple M4 芯片
  - CPU推理速度：**85.69 tokens/秒**（非常快！）
  - 意外发现：对于小模型（104M），CPU 比 MPS (GPU) 快 2 倍
  - 原因：数据传输开销 > 计算收益，M4 CPU 性能极强

---

### 2025-11-07：深度理解 Transformer 核心组件

#### ✅ 完成事项
- [x] 选择学习路径 A（代码架构深度学习）
- [x] 理解为什么需要归一化（梯度消失/爆炸问题）
- [x] 理解 RMSNorm 的工作原理
- [x] 对比 LayerNorm 和 RMSNorm
- [x] 理解 RoPE 位置编码的基本原理
- [x] 理解 RoPE 的多频率机制
- [x] 创建学习辅助材料文件夹结构
- [x] 更新 CLAUDE.md（添加互动学习方式）
- [x] 拆分 notes 为学习日志和知识库

#### 💭 个人思考

**2025-11-07：第一次深入理解归一化**
- **收获**：终于理解了为什么需要归一化！不是"玄学"，而是真的会梯度消失
- **惊喜**：RMSNorm 比 LayerNorm 快这么多！（7.7 倍）
- **理解加深**：归一化不是丢失信息，只是控制数值大小，信息还在
- **新认知**：RMSNorm 不是单独一层，而是 Transformer Block 的组件
- **学习方式调整**：慢下来，一个概念一个概念吃透，不着急

**2025-11-07：第一次深入理解 RoPE**
- **收获**：理解了 Attention 的"排列不变性"问题
- **惊喜**：RoPE 用旋转来编码位置，太优雅了！
- **新认知**：
  - RoPE 同时包含绝对和相对位置信息（两全其美）
  - 多频率机制避免了"转一圈回到原点"的问题
  - 32个频率组合可以编码百万级别的位置
- **疑问解答**：
  - Q: 旋转720度不是回到原点了吗？
  - A: 使用多频率！就像钟表有时针、分针、秒针
  - Q: 绝对位置丢失了吗？
  - A: 没有！每个词被旋转到特定角度，还是知道绝对位置的

**2025-11-07：深入理解多频率的必要性** ⭐️
- **关键问题**：为什么需要 32 个频率？只用一个超低频率不行吗？
- **重大发现**：浮点数精度限制！
  - 理论上：一个超低频率能覆盖所有位置 ✅
  - 实践中：float32 精度不够，相邻位置无法区分 ❌
  - 实验证据：位置0和1的 cos 值在 float32 下完全相同（都是 1.0）
- **数学本质**：
  - 超低频率的相邻位置差 ≈ 10^-11
  - float32 精度下限 ≈ 10^-7
  - 差值 < 精度 → 无法表示！
- **解决方案**：
  - 高频率（频率0）：相邻位置差 57.3°（远超精度）
  - 低频率（频率31）：覆盖 600 万个 token
  - 组合：完美平衡精度和覆盖范围
- **类比理解**：
  - 用 1cm 刻度的尺子测量 0.01mm → 测不出来
  - 用显微镜+望远镜组合 → 既看清细节又看得远
- **核心认知**：RoPE 多频率是**数学理论 + 计算机硬件约束**的完美结合

---

## 📚 学习资源

### 代码文件位置
- 主要实现：`model/model_minimind.py`（471 行）
- 学习材料：`learning_materials/`（7 个示例程序）
- RMSNorm 实现：`model/model_minimind.py:95-105`
- RoPE 实现：`model/model_minimind.py:108-137`

### 学习辅助材料

**归一化相关**：
- `learning_materials/why_normalization.py` - 演示梯度消失问题
- `learning_materials/rmsnorm_explained.py` - RMSNorm 原理和效果
- `learning_materials/normalization_comparison.py` - LayerNorm vs RMSNorm 对比

**位置编码相关**：
- `learning_materials/rope_basics.py` - RoPE 基础原理
- `learning_materials/rope_multi_frequency.py` - 多频率机制详解
- `learning_materials/rope_why_multi_frequency.py` - 为什么需要多频率（浮点数精度问题）
- `learning_materials/rope_explained.py` - 完整实现（高级）

**注意力机制相关**：
- `learning_materials/attention_explained.py` - Multi-Head Attention（待学习）

**使用说明**：
- `learning_materials/README.md` - 完整使用指南

### 推荐学习顺序
1. ✅ RMSNorm 归一化机制
2. ✅ RoPE 位置编码
3. ⏳ Attention 注意力机制（下一步）
4. ⏳ FeedForward 前馈网络
5. ⏳ 完整的 Transformer Block

---

### 2025-11-10: 深入理解 Attention 注意力机制

#### ✅ 完成事项
- [x] 理解 Attention 的核心概念（词与词的相关性）
- [x] 理解 Self-Attention vs Cross-Attention
- [x] 理解 Q、K、V 的作用和数据库查询类比
- [x] 理解权重矩阵 W_Q、W_K、W_V 是模型参数
- [x] 理解 Attention 计算流程（scores → softmax → weighted sum）
- [x] 理解 Multi-Head Attention 的多面性
- [x] 理解每个头的维度计算（head_dim = hidden_size / num_heads）
- [x] 理解多头输出的合并方式（reshape 拼接）
- [x] 理解 RoPE 在 Attention 中的应用位置
- [x] 创建 Q、K、V 学习示例代码（attention_qkv_explained.py）
- [x] 更新 knowledge_base.md 完整的 Attention 章节
- [x] 理解 Softmax 和 RMSNorm 的区别和位置
- [x] 理解 FeedForward 的核心作用（非线性变换）
- [x] 理解"扩张-压缩"机制（768 → 2048 → 768）
- [x] 理解 SwiGLU 激活函数（gate × up）
- [x] 理解 Attention vs FeedForward 的分工
- [x] 创建 FeedForward 学习示例代码（feedforward_explained.py）
- [x] 更新 knowledge_base.md 完整的 FeedForward 章节
- [x] 理解 Transformer Block 的组装方式
- [x] 理解残差连接的作用（保底机制 + 梯度高速公路）
- [x] 理解 Pre-Norm vs Post-Norm 的区别
- [x] 理解 MiniMindBlock 的完整数据流
- [x] 思考学习路径（实践 vs 原理）

#### 💭 个人思考

**2025-11-10：Attention 学习感受**
- **理解突破**：用数据库查询的类比理解 Q、K、V，太清晰了！
- **关键认知**：
  - Q、K、V 不是三种不同的东西，而是同一个输入的三个"视角"
  - 权重矩阵 W_Q、W_K、W_V 是训练出来的参数，不是固定的
  - Multi-Head 的"多面性"不是增加 Q、K、V 数量，而是增加 Head 数量
  - RoPE 只用于 Q、K（计算相似度），不用于 V（内容）
- **学习方式调整**：一开始觉得很复杂，但把整体架构理清楚后，每个细节都能对上号了
- **新的疑问**：
  - Causal Mask 是什么？（看到代码里有）
  - GQA (Grouped Query Attention) 是什么？（配置里提到）
  - KV Cache 是什么优化？

**2025-11-10：Multi-Head Attention 的精髓**
- **核心理解**：就像用 8 副不同的眼镜看同一句话
  - 眼镜1：看语法
  - 眼镜2：看语义
  - 眼镜3：看情感
  - ...
  - 最后把 8 个视角融合在一起
- **数学本质**：
  - 拆分：768 维 → 8 个 96 维
  - 并行计算：8 个头独立做 Attention
  - 合并：8 × 96 = 768（恢复原始维度）
- **不变量**：输入维度 = 输出维度 = 768

**2025-11-10：FeedForward 前馈网络的理解**
- **初始困惑**："扩张-压缩"到底在做什么？为什么不直接 768 → 768？
- **关键突破**：
  - 直接 768 → 768：只是线性变换，表达能力有限
  - 768 → 2048 → 768：经过高维空间，能做复杂的非线性变换
  - 类比：做菜（食材→加工→装盘），照片处理（像素→特征提取→优化像素）
- **核心认知**：
  - FeedForward 每个词独立处理（vs Attention 词与词交互）
  - Attention = 开会讨论，FeedForward = 各自思考
  - SwiGLU 比普通 FFN 更强：门控机制（gate × up）+ SiLU 激活
- **实现细节**：
  - gate_proj: 768 → 2048（门控分支）
  - up_proj: 768 → 2048（上投影分支）
  - hidden = SiLU(gate) * up（逐元素相乘）
  - down_proj: 2048 → 768（压缩回原维度）
- **与 Attention 的分工**：
  - Attention：让模型知道"哪些词相关"
  - FeedForward：让模型知道"如何处理这些信息"
  - 两者缺一不可！
- **创建学习材料**：feedforward_explained.py

**2025-11-10：Transformer Block 组装理解**
- **核心问题**：4 个组件如何组合成一个完整的"思考单元"？
- **关键概念 - 残差连接** ⭐⭐⭐
  - **初始困惑**：为什么要 `hidden_states += residual`？
  - **照片修图类比**：
    - 没有残差：每一步完全覆盖前一步（危险）
    - 有残差：原始 + 所有调整的累积效果（安全）
  - **数学本质**：
    - `y = x + F(x)` → `dy/dx = 1 + dF/dx`
    - 即使 F 学不到东西（dF/dx → 0），至少还有 1（梯度能传回）
  - **作用**：
    - 保底机制：最坏情况输出 = 输入
    - 增量学习：只需学"在输入基础上调整什么"
    - 梯度高速公路：梯度可以跳过中间层直接传回
- **数据流理解**：
  ```
  输入 x
    ├─ 保存 residual
    ↓
  RMSNorm #1
    ↓
  Attention (+ RoPE)
    ↓
  + residual ← 第一个残差连接
    ├─ 保存当前状态
    ↓
  RMSNorm #2
    ↓
  FeedForward
    ↓
  + residual ← 第二个残差连接
    ↓
  输出
  ```
- **Pre-Norm vs Post-Norm**：
  - MiniMind 使用 Pre-Norm（归一化在子层之前）
  - Pre-Norm 更稳定，适合深层网络（>12层）
  - 残差路径更"干净"（不被 Norm 打断）
- **代码位置**：model/model_minimind.py:359-380

**2025-11-10：学习路径的思考**
- **关键问题**：原仓库能学到这些知识吗？
- **答案**：可以，但需要自己"挖掘"
  - 原仓库提供：完整的代码实现 + 简单的说明
  - 需要自己：阅读代码 + 查外部资料 + 推导数学 + 写示例验证
- **两种学习路径**：
  - 路径 A（实践导向）：2 小时训练模型，学"怎么用"
  - 路径 B（原理导向）：几周深度理解，学"为什么"
- **我的选择**：原理导向，把隐性知识显性化
- **收获**：理解更深、记忆更牢、可迁移、可创新

---

## 🎯 下次学习计划

**当前进度**：Transformer 架构学习接近完成 ✨
- ✅ RMSNorm（归一化）- 11-07 完成
- ✅ RoPE（位置编码）- 11-07 完成
- ✅ Attention（注意力机制）- 11-10 完成
- ✅ FeedForward（前馈网络）- 11-10 完成
- ✅ Transformer Block（组装）- 11-10 完成
- ⏳ 整体 Transformer 架构（90% 完成）

**下次学习**：
- [ ] 完整的 MiniMindModel 架构
  - 词嵌入层（embed_tokens）
  - N 个 Transformer Block 的堆叠
  - 最终的 RMSNorm
  - 输出层（lm_head）
  - 自回归生成流程

**可选深入**（以后有时间再学）：
- GQA (Grouped Query Attention)
- Causal Mask（因果掩码）
- KV Cache（推理优化）
- Flash Attention

**硬件配置记录**：
- CPU: Apple M4（性能强劲）
- 内存: 16 GB（足够训练小模型）
- 可用空间: 2.7 GB（偏紧张，暂不训练）
- 加速: MPS 可用（但小模型用 CPU 更快）

---

**最后更新**：2025-11-10
**学习进度**：第一阶段 - Transformer 架构学习接近完成（90%）

**今日收获总结**（2025-11-10）：
- 完成了 4 个核心组件的深度学习（Attention + FeedForward + Block 组装）
- 理解了残差连接的精髓：保底机制 + 增量学习 + 梯度高速公路
- 理解了 Pre-Norm 的优势：训练更稳定，适合深层网络
- 思考了学习路径：选择原理导向，追求深度理解
- 创建了 2 个学习材料：attention_qkv_explained.py, feedforward_explained.py
- 更新了完整的 knowledge_base.md：FeedForward 章节 + Transformer 架构章节
