---
title: 学习笔记总索引 | minimind从零理解llm训练
description: MiniMind 学习笔记系统的导航入口。包含学习日志、知识库和学习辅助材料的完整索引，帮助快速查找学习资源。
keywords: LLM学习笔记, 学习笔记索引, MiniMind笔记, Transformer学习笔记, 深度学习笔记
---

# MiniMind 学习笔记 - 总索引

> 这是我学习 MiniMind 项目的笔记系统入口

---

## 📚 文档结构

为了方便复习和查找，笔记系统分为三个文件：

### 1. **learning_log.md** - 学习日志
📝 记录每日学习进度、遇到的问题和解决方案

**内容**：
- 每日完成事项
- 遇到的Bug和解决方案
- 个人思考和感悟
- 学习计划

**查看**：[learning_log.md](./learning_log.md)

---

### 2. **knowledge_base.md** - 知识库
📖 系统化整理的知识点、概念解释和问答记录

**内容**：
- 归一化机制（RMSNorm, LayerNorm）
- 位置编码（RoPE）
- Transformer 架构
- 问答记录（Q&A）
- 重要公式汇总

**查看**：[knowledge_base.md](./knowledge_base.md)

---

### 3. **learning_materials/** - 学习辅助材料
💻 可执行的示例代码，帮助理解核心概念

**内容**：
- 归一化相关（3个程序）
- 位置编码相关（3个程序）
- 注意力机制相关（1个程序）
- README 使用说明

**查看**：[learning_materials/README.md](./learning_materials/README.md)

---

## 🔍 快速查找

### 按主题查找

| 主题 | 查看位置 |
|------|----------|
| 环境搭建问题 | [learning_log.md § 2025-11-06](./learning_log.md#2025-11-06环境搭建--首次运行模型--性能测试) |
| RMSNorm 原理 | [knowledge_base.md § 归一化机制](./knowledge_base.md#1-归一化机制) |
| RoPE 原理 | [knowledge_base.md § 位置编码](./knowledge_base.md#2-位置编码) |
| 问题解答 | [knowledge_base.md § 问答记录](./knowledge_base.md#问答记录) |
| 可执行示例 | [learning_materials/README.md](./learning_materials/README.md) |

### 按日期查找

- **2025-11-06**: [环境搭建 + 模型运行](./learning_log.md#2025-11-06环境搭建--首次运行模型--性能测试)
- **2025-11-07**: [Transformer 核心组件学习](./learning_log.md#2025-11-07深度理解-transformer-核心组件)
- **2025-11-10**: [Attention + FeedForward + Transformer Block](./learning_log.md#2025-11-10-深入理解-attention-注意力机制)
- **2025-12-27**: [确认 Tier 1 完成](./learning_log.md#2025-12-27完成-tier-1-基础组件学习)
- **2026-01-18**: [深入理解完整模型架构](./learning_log.md#2026-01-18深入理解完整模型架构)
- **2026-05-25**: [理解残差连接 + Tier 2 教学模块](./learning_log.md#2026-05-25-理解残差连接--创建-tier-2-教学模块)

---

## 🎯 当前学习进度

**阶段**：Tier 1 完成 + Tier 2 进行中

**Tier 1 - Foundation（基础组件）**：✅ 100%
- ✅ RMSNorm（归一化）
- ✅ RoPE（位置编码）
- ✅ Attention（注意力机制）
- ✅ FeedForward（前馈网络）
- ✅ Transformer Block（组装）

**Tier 2 - Architecture（架构组装）**：🚧 进行中
- ✅ 01. Residual Connection（残差连接）
- 📋 02. Transformer Block（待创建）

**完整模型架构**：✅ 100%
- ✅ MiniMindForCausalLM（接口层）
- ✅ MiniMindModel（核心层）
- ✅ 词嵌入层（embed_tokens）
- ✅ 8 层 Transformer Block 堆叠
- ✅ 输出层（lm_head）
- ✅ 权重共享（Weight Tying）
- ✅ 完整数据流

**下一步**：
- 创建 02. Transformer Block 教学模块
- 理解 Pre-Norm Block 组件编排顺序
- 自回归生成机制
- KV Cache 推理优化

---

## 📂 项目文件结构

```
minimind/
├── notes.md                    ← 你现在看的（总索引）
├── learning_log.md             ← 学习日志
├── knowledge_base.md           ← 知识库
├── learning_materials/         ← 学习辅助材料
│   ├── README.md
│   ├── why_normalization.py
│   ├── rmsnorm_explained.py
│   ├── normalization_comparison.py
│   ├── rope_basics.py
│   ├── rope_multi_frequency.py
│   ├── rope_explained.py
│   └── attention_explained.py
├── CLAUDE.md                   ← AI 助手指南
├── model/                      ← 真实代码实现
│   └── model_minimind.py
├── trainer/                    ← 训练脚本
├── dataset/                    ← 数据集
└── MiniMind2/                  ← 预训练模型
```

---

## 🚀 快速启动命令

```bash
# 进入项目
cd /Users/de-shiouhuang/Documents/code/minimind

# 激活虚拟环境（每次使用前必须执行）
source venv/bin/activate

# 测试预训练模型
python eval_llm.py --load_from ./MiniMind2

# 运行学习材料
python learning_materials/why_normalization.py
python learning_materials/rope_basics.py
```

---

## 💡 使用建议

1. **学习新知识时**：
   - 先运行 `learning_materials/` 中的相关程序
   - 查看 `knowledge_base.md` 中的系统化解释
   - 在 `learning_log.md` 中记录个人思考

2. **遇到问题时**：
   - 查看 `learning_log.md` 中的问题解决方案
   - 查看 `knowledge_base.md` 中的问答记录

3. **复习时**：
   - 按主题查找 `knowledge_base.md`
   - 按日期回顾 `learning_log.md`

---

**最后更新**：2026-05-25
**学习进度**：Tier 1 完成 ✅ + Tier 2 开始 🚧
