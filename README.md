# MiniMind 学习笔记

> 本仓库是我个人学习 [MiniMind](https://github.com/jingyaogong/minimind) 项目的学习资料及源码备份

## 📚 关于本仓库

这是我从零开始学习大语言模型（LLM）的完整记录，基于 MiniMind 开源项目进行深度学习。

**MiniMind** 是一个教育性质的 LLM 训练项目，特点：
- 仅需 3 块钱成本 + 2 小时即可训练 25.8M 参数的超小语言模型
- 完整开源：预训练、SFT、LoRA、DPO、RLAIF(PPO/GRPO)、模型蒸馏等全流程
- 所有核心算法从零使用 PyTorch 实现，不依赖第三方抽象接口
- 适合入门 LLM 原理和实践

## 📖 学习笔记系统

本仓库采用三层笔记结构：

```
minimind-notes/
├── notes.md                ← 📌 总索引（从这里开始）
├── learning_log.md         ← 📝 学习日志（按时间顺序）
├── knowledge_base.md       ← 📚 知识库（按主题整理）
└── learning_materials/     ← 💻 可执行示例代码
```

### 快速导航

- **[notes.md](./notes.md)** - 总索引和快速查找入口
- **[learning_log.md](./learning_log.md)** - 每日学习进度和思考
- **[knowledge_base.md](./knowledge_base.md)** - 系统化的知识点整理
- **[learning_materials/](./learning_materials/)** - 手写示例代码

## 🎯 学习进度

**当前阶段**：Transformer 核心组件学习

- ✅ 环境搭建与模型运行
- ✅ RMSNorm（归一化机制）
- ✅ RoPE（旋转位置编码）
- ⏳ Attention（注意力机制）- 进行中
- ⏳ FeedForward（前馈网络）
- ⏳ 完整的 Transformer Block

## 🔧 项目结构

```
minimind-notes/
├── model/                    # 模型实现
│   ├── model_minimind.py    # MiniMind 核心代码
│   └── model_lora.py        # LoRA 实现
├── trainer/                  # 训练脚本
│   ├── train_pretrain.py    # 预训练
│   ├── train_full_sft.py    # 监督微调
│   ├── train_dpo.py         # DPO（RLHF）
│   └── ...                  # 其他训练阶段
├── dataset/                  # 数据集处理
├── scripts/                  # 工具脚本
└── eval_llm.py              # 模型评测和对话
```

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/joyehuang/minimind-notes.git
cd minimind-notes

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型（可选）
git clone https://huggingface.co/jingyaogong/MiniMind2

# 运行模型对话
python eval_llm.py --load_from ./MiniMind2

# 运行学习材料
python learning_materials/rmsnorm_explained.py
python learning_materials/rope_basics.py
```

## 📝 学习方法

1. **理论学习**：阅读 `knowledge_base.md` 中的系统化知识
2. **实践验证**：运行 `learning_materials/` 中的示例代码
3. **记录思考**：在 `learning_log.md` 中写下每日收获
4. **深入源码**：对照 `model/model_minimind.py` 理解实现细节

## 🔗 相关资源

- **原项目**：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **模型权重**：[HuggingFace Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **数据集**：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset)

## 📄 许可证

本仓库的学习笔记部分遵循 CC BY 4.0 许可。

MiniMind 源代码部分保留原项目的 Apache License 2.0 许可。

## 🙏 致谢

感谢 [MiniMind](https://github.com/jingyaogong/minimind) 项目提供了如此优秀的学习资源！

---

**学习者**：joyehuang
**开始时间**：2025-11-06
**当前进度**：Transformer 核心组件学习中（2/4 完成）
