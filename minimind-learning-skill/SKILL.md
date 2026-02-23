---
name: minimind-learning
description: "MiniMind 学习助手。自动记录学习笔记，识别 RMSNorm, LayerNorm, RoPE, Attention, LoRA, DPO, PPO, GRPO, SFT, RLHF 等术语。触发词：学习、开始、MiniMind、归一化、位置编码、注意力、训练、微调。"
metadata:
  author: Joye Huang
  version: "1.0.0"
  license: MIT
  tags: minimind, learning, llm, transformer, training
---

# MiniMind Learning Assistant

自动化学习笔记系统，静默记录你的 MiniMind 学习历程。

## When to use

**自动激活场景**：

1. **学习开始时**：
   - 用户说："开始今天的学习"、"开始学习"、"今天学什么"
   - 用户说："继续学习"、"学习 MiniMind"

2. **讨论 MiniMind 内容时**：
   - 提问包含：RMSNorm, LayerNorm, RoPE, YaRN, Attention, GQA, SwiGLU, Transformer, LoRA, DPO, PPO, GRPO, SFT, RLHF, RLAIF, MoE
   - 问题词：什么是、如何、为什么、怎样、解释、原理
   - 遇到问题：报错、错误、失败、Bug

3. **显式记录请求**：
   - 用户说："记录一下"、"保存笔记"、"写入笔记"

## How to use

### 1. 初始化检查

首次激活时，确保笔记结构存在：

```bash
# 检测 Git 仓库根目录
git rev-parse --show-toplevel

# 验证是 MiniMind 仓库（存在以下文件）
# - model/model_minimind.py
# - trainer/train_pretrain.py
# - README.md (包含 "MiniMind")

# 创建笔记目录（如不存在）
mkdir -p docs/learning_materials

# 从模板初始化文件（如不存在）
# - docs/notes.md
# - docs/learning_log.md
# - docs/knowledge_base.md
# - docs/learning_materials/README.md
```

**模板位置**：`~/.claude/skills/minimind-learning/templates/`

### 2. 学习引导模式

**当用户说"开始学习"时**，主动引导：

```markdown
👋 欢迎开始今天的 MiniMind 学习！

你想学习哪个模块？

**基础组件**：
1. 归一化技术 - RMSNorm, LayerNorm
2. 位置编码 - RoPE, YaRN
3. 注意力机制 - Attention, GQA
4. 前馈网络 - FeedForward, SwiGLU

**训练技术**：
5. 预训练 - Pretraining
6. 监督微调 - SFT
7. 参数高效微调 - LoRA
8. 强化学习 - DPO, PPO, GRPO

直接告诉我编号或主题名称，我会为你讲解！

（学习过程中我会自动记录笔记到 `docs/` 目录）
```

### 3. 触发检测

**每次用户消息后**，检查是否满足以下任一条件：

#### Tier 1: 即时触发（立即更新笔记）

```python
# MiniMind 术语（50+）
TERMS = [
    # 架构
    "RMSNorm", "LayerNorm", "BatchNorm", "GroupNorm",
    "RoPE", "YaRN", "ALiBi", "位置编码",
    "Attention", "注意力", "GQA", "MQA", "FlashAttention",
    "FeedForward", "前馈", "SwiGLU", "GELU", "GLU",
    "Transformer", "TransformerBlock",

    # 训练
    "pretrain", "预训练", "pretraining",
    "SFT", "监督微调", "fine-tuning", "微调",
    "LoRA", "lora", "LoRA-r", "LoRA-alpha",
    "DPO", "PPO", "GRPO", "SPO",
    "RLHF", "RLAIF", "强化学习",
    "distillation", "蒸馏", "知识蒸馏",

    # 模型
    "MiniMind", "MiniMind-Dense", "MiniMind-MoE",
    "MoE", "混合专家", "expert routing",
    "MiniMind-Reason", "R1",
]

# 问题词
QUESTION_WORDS = ["什么是", "如何", "为什么", "怎样", "解释", "原理", "作用"]

# 问题指示
PROBLEM_MARKERS = ["报错", "错误", "失败", "Bug", "不工作", "问题"]

# 检查
if any(term in user_message for term in TERMS):
    trigger_tier_1()
elif any(word in user_message for word in QUESTION_WORDS):
    trigger_tier_1()
elif any(marker in user_message for marker in PROBLEM_MARKERS):
    trigger_tier_1()
```

#### Tier 2: 延迟触发（5秒后批量更新）

```python
# 深度对话特征
if (
    conversation_turns >= 3 or  # 多轮对话
    "```python" in assistant_response or  # 包含代码
    "$" in assistant_response or  # 包含公式
    len(assistant_response) > 1000 or  # 长回复
    "model/" in assistant_response or  # 引用源码
    "trainer/" in assistant_response
):
    trigger_tier_2()  # 延迟 5 秒
```

#### Tier 3: 显式触发（总是更新）

```python
EXPLICIT_KEYWORDS = ["记录", "记下", "保存", "写入笔记", "更新笔记"]

if any(kw in user_message for kw in EXPLICIT_KEYWORDS):
    trigger_tier_3()
```

### 4. 内容提取

**从对话中提取结构化信息**：

```python
# 提取问题
def extract_question(user_message):
    patterns = [
        r"^(.*[?？])$",  # 问号结尾
        r"^(什么是|如何|为什么)(.*?)([?？。]|$)",
        r"(.*)(吗|呢)[?？。]*$"
    ]
    # 返回匹配的问题

# 提取概念定义
def extract_concepts(assistant_response):
    patterns = [
        r"([A-Z\u4e00-\u9fa5]{2,})\s*(是|：)(.*?)([。\n]|$)",
        r"\*\*([^*]+)\*\*\s*[：:](.*?)([。\n]|$)",
        r"###\s+([^\n]+)\n\n([^\n]+)"
    ]
    # 返回 [(概念, 定义), ...]

# 提取问题解决方案
def extract_problem_solution(conversation):
    problem = {
        "description": "",  # 错误现象
        "root_cause": "",   # 根本原因
        "solution": ""      # 解决方案
    }
    # 解析对话提取信息

# 提取代码示例
def extract_code_blocks(response):
    return re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
```

### 5. 文件更新

**更新 learning_log.md**：

```python
def update_learning_log(date, topic, tasks, problems, reflections, materials):
    """
    格式:
    ### 2026-02-23: 理解 RoPE 多频率机制

    #### ✅ 完成事项
    - [x] 理解为什么需要多频率

    #### 🐛 遇到的问题
    **问题: ...**
    - **错误现象**: ...
    - **根本原因**: ...
    - **解决方案**: ...

    #### 💭 个人思考
    - **收获**: ...

    #### 📝 相关学习材料
    - 新增代码: `learning_materials/xxx.py`
    """

    # 检查今天日期章节是否存在
    if f"### {date}:" in content:
        # 追加子章节
        insert_subsection_after_date(date, new_content)
    else:
        # 插入新日期章节（按时间倒序）
        insert_date_section_chronologically(date, new_content)
```

**更新 knowledge_base.md**：

```python
def update_knowledge_base(question, answer, details, code_example):
    """
    格式:
    **Q20: 为什么 RoPE 需要多频率？** [⭐️]

    A: 因为单一低频率受浮点数精度限制。

    **详细说明**:
    - 详细解释1
    - 详细解释2

    **代码示例**:
    ```python
    # 代码
    ```

    参考代码: `learning_materials/xxx.py`

    ---
    """

    # 1. 找到最大 Q 编号
    existing_q = re.findall(r"Q(\d+)", content)
    next_q = max(existing_q) + 1 if existing_q else 1

    # 2. 推断所属章节
    category = infer_category(question)
    # "归一化" / "位置编码" / "注意力" / "训练" 等

    # 3. 在章节末尾插入
    insert_at_category_end(category, qa_entry)

    # 4. 标记重要问题
    if any(kw in question for kw in ["原理", "为什么", "核心", "本质"]):
        mark_with_star(qa_entry)
```

**更新 learning_materials/README.md**：

```python
def update_materials_readme(new_file):
    """
    格式:
    ## 位置编码

    - **`rope_multi_freq.py`** - 多频率机制验证
      - 验证浮点数精度限制
      - 对比单频率 vs 多频率
    """

    # 从文件提取描述（docstring）
    description = extract_file_description(new_file)

    # 推断分类
    category = infer_category_from_filename(new_file)

    # 插入条目
    insert_at_category_end(category, entry)
```

### 6. Git 自动化

**生成 Commit Message**：

```python
def generate_commit_message(changes):
    """
    模式: "[动作] [主题] [子主题]"

    动作词:
    - 学习 (新概念)
    - 理解 (深入理解)
    - 添加 (代码/材料)
    - 解决 (问题)
    - 完善 (补充)
    """

    # 提取主要 MiniMind 术语
    primary_term = extract_primary_term(changes.content)

    # 识别动作类型
    action = identify_action(changes)

    # 提取子主题
    sub_topic = extract_sub_topic(changes.content)

    # 构造 message
    message = f"{action} {primary_term}"
    if sub_topic:
        message += f" {sub_topic}"

    return message[:30]  # 限制长度

# 示例输出:
# "学习 RMSNorm 归一化原理"
# "理解 RoPE 多频率机制"
# "添加 Attention 学习材料"
# "解决 CUDA 内存溢出问题"
```

**执行 Git 操作**：

```bash
# 自动执行（静默）
cd {repo_root}
git add docs/notes.md docs/learning_log.md docs/knowledge_base.md docs/learning_materials/
git commit -m "{generated_message}"
git push origin {current_branch}
```

**错误处理**：

```python
def safe_git_push(max_retries=3):
    for attempt in range(max_retries):
        try:
            result = run_git_push(timeout=30)
            if result.success:
                return True
        except TimeoutError:
            wait = 2 ** attempt  # 指数退避: 1s, 2s, 4s
            sleep(wait)

    # 失败后不阻塞，记录日志
    log_warning("Git push 失败，更改已提交到本地")
    return False
```

### 7. 分类推断

**自动推断知识点所属主题**：

```python
CATEGORY_KEYWORDS = {
    "归一化技术": ["归一化", "Norm", "RMS", "Layer", "Batch", "Group"],
    "位置编码": ["位置", "RoPE", "YaRN", "编码", "位置编码", "ALiBi"],
    "注意力机制": ["注意力", "Attention", "GQA", "MQA", "FlashAttention"],
    "前馈网络": ["前馈", "FeedForward", "SwiGLU", "GLU", "GELU"],
    "预训练": ["预训练", "pretrain", "pretraining", "语言模型"],
    "监督微调": ["SFT", "微调", "fine-tuning", "监督"],
    "参数高效微调": ["LoRA", "lora", "PEFT", "参数高效"],
    "人类反馈强化学习": ["DPO", "PPO", "GRPO", "RLHF", "RLAIF", "强化学习"],
    "Transformer 架构": ["Transformer", "架构", "模型结构"],
    "混合专家模型": ["MoE", "混合专家", "expert", "routing"],
}

def infer_category(text):
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return "其他"  # 默认分类
```

### 8. 静默运行原则

**不打扰用户**：

```python
# ❌ 不要这样
print("正在更新笔记...")
print("已保存到 learning_log.md")
print("Git 提交成功")

# ✅ 应该这样
# 完全静默，只在出错时提示
if git_push_failed:
    # 仅在失败时简短提示
    print("💡 提示：更改已保存到本地，推送失败（网络问题）")
```

**专注学习对话**：

```python
# ✅ 继续正常对话
user: "什么是 RMSNorm？"
assistant: "RMSNorm (Root Mean Square Normalization) 是..."
# [背后静默更新笔记]

user: "它和 LayerNorm 有什么区别？"
assistant: "主要区别有三点..."
# [背后再次更新]

# 用户完全不知道笔记在更新
```

### 9. 模板文件

**初始化时使用的模板**：

```bash
~/.claude/skills/minimind-learning/templates/
├── notes.md.template
├── learning_log.md.template
├── knowledge_base.md.template
└── learning_materials_readme.md.template
```

**模板变量替换**：

```python
def load_template(template_name):
    template_path = Path.home() / ".claude/skills/minimind-learning/templates" / template_name
    content = template_path.read_text(encoding="utf-8")

    # 替换占位符
    today = datetime.now().strftime("%Y-%m-%d")
    content = content.replace("{TODAY}", today)

    return content
```

### 10. 配置读取（可选）

**如果存在 `.minimind-learning.json`**：

```python
config_path = repo_root / ".minimind-learning.json"
if config_path.exists():
    config = json.loads(config_path.read_text())

    auto_commit = config.get("auto_commit", True)
    auto_push = config.get("auto_push", True)
    batch_delay = config.get("batch_delay", 5)
    notes_dir = config.get("notes_dir", "docs")
    mark_important = config.get("mark_important", True)
```

## Best Practices

### 引导式学习

当用户说"开始学习"时，主动提供学习路径：

```markdown
📚 MiniMind 推荐学习路径:

**Week 1: 基础组件**
→ Day 1-2: 归一化技术 (RMSNorm)
→ Day 3-4: 位置编码 (RoPE)
→ Day 5-7: 注意力机制 (Attention, GQA)

**Week 2: 完整架构**
→ Day 8-10: Transformer Block
→ Day 11-14: 完整模型实现

**Week 3-4: 训练技术**
→ 预训练 → SFT → LoRA → RLHF

你想从哪里开始？
```

### 鼓励实践

检测到代码讨论时，建议创建可执行示例：

```markdown
💡 要不要创建一个可运行的代码示例？

我可以帮你创建 `learning_materials/rope_basics.py`，
包含完整的 RoPE 实现和可视化。

这样你可以直接运行看效果！
```

### 定期总结

检测到学习了多个知识点后，主动总结：

```markdown
📊 今天学习总结:

✅ 完成事项:
- 理解了 RMSNorm 的原理
- 对比了 RMSNorm vs LayerNorm
- 运行了验证代码

🎯 建议:
明天可以学习 RoPE 位置编码，它和 RMSNorm
一起构成了现代 Transformer 的基础。

（笔记已自动保存到 `docs/learning_log.md`）
```

## Error Handling

### 仓库检测失败

```python
if not is_minimind_repo():
    print("❌ 当前目录不是 MiniMind 仓库")
    print("请确保在 MiniMind 目录中使用此 skill")
    print("或检查以下文件是否存在:")
    print("  - model/model_minimind.py")
    print("  - trainer/train_pretrain.py")
    return
```

### Git 操作失败

```python
if git_commit_failed:
    print("⚠️  Git 提交失败，但笔记已更新")
    print("请手动提交:")
    print("  cd docs/")
    print("  git add .")
    print("  git commit -m '学习笔记更新'")
```

### 文件冲突

```python
if file_conflict_detected:
    print("⚠️  检测到文件冲突")
    print("建议:")
    print("1. 手动解决冲突")
    print("2. 或运行验证脚本:")
    print("   python ~/.claude/skills/minimind-learning/scripts/validate_notes.py --fix-numbering")
```

## Validation

**定期提醒用户验证笔记**（每 10 个 Q&A 后）：

```markdown
💡 笔记提示:

你已经积累了 10 个问答！建议运行验证脚本:

```bash
python ~/.claude/skills/minimind-learning/scripts/validate_notes.py
```

这会检查:
- Q 编号连续性
- 日期格式
- 文件引用完整性
```

## Summary

**这个 skill 的核心行为**：

1. ✅ **静默监听**：每次对话后检查触发条件
2. ✅ **智能提取**：从对话中提取问题、概念、代码
3. ✅ **自动更新**：更新三套笔记文件
4. ✅ **Git 自动化**：生成简洁 commit 并推送
5. ✅ **主动引导**：提供学习路径和建议

**用户体验**：
- 说"开始学习" → 立即得到学习指引
- 自然提问 → 背后自动记录笔记
- 完全静默 → 专注学习，无打扰
- 定期总结 → 巩固学习成果

---

**版本**: 1.0.0
**作者**: Joye Huang
**许可**: MIT
