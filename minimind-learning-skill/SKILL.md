# MiniMind Learning Assistant

A specialized Claude Code skill that automatically maintains learning notes for MiniMind LLM training framework learners. This skill silently records your learning journey through conversational dialogue.

---
name: minimind-learning
version: 1.0.0
author: Joye Huang
repository: https://github.com/joyehuang/minimind-notes
license: MIT
---

## Overview

This skill transforms your Claude Code conversations about MiniMind into structured learning notes, automatically detecting important learning moments and maintaining a comprehensive knowledge base.

**Key Features**:
- 🤖 **Silent Operation**: Updates notes automatically without interrupting your flow
- 🧠 **Context-Aware**: Deep understanding of 50+ MiniMind-specific terms (RMSNorm, RoPE, DPO, LoRA, etc.)
- 📚 **Three-Tier Note System**: learning_log.md (chronological), knowledge_base.md (topical), learning_materials/ (code examples)
- 🔄 **Full Git Automation**: Commits and pushes with clean, concise Chinese messages
- 🎯 **Smart Triggering**: Three-layer detection system (keywords, conversation depth, explicit requests)

## How It Works

### Automatic Triggering System

**Tier 1: Immediate Triggers** (Update within 2 seconds)
- MiniMind terminology detected: RMSNorm, LayerNorm, RoPE, YaRN, Attention, GQA, SwiGLU, Transformer, LoRA, DPO, PPO, GRPO, SPO, SFT, RLHF, RLAIF, MoE, distillation
- Question words: 什么是, 如何, 为什么, 怎样, 解释, 原理, 作用
- Problem indicators: 报错, 错误, 问题, 失败, Bug

**Tier 2: Delayed Triggers** (Batch update after 5 seconds of inactivity)
- Multi-turn conversations (3+ exchanges)
- Contains code blocks (```python)
- Contains mathematical formulas ($...$)
- Long responses (>1000 characters)
- References source files (model/*.py, trainer/*.py)

**Tier 3: Explicit Requests** (Always triggers)
- User says: 记录, 记下, 保存, 写入笔记, 更新笔记

### File Update Logic

#### learning_log.md - Chronological Journal
```markdown
### 2026-02-23: 理解 RoPE 多频率机制

#### ✅ 完成事项
- [x] 理解为什么需要多频率
- [x] 理解浮点数精度限制

#### 🐛 遇到的问题
**问题: 单一低频率不够？**
- **错误现象**: ...
- **根本原因**: ...
- **解决方案**: ...

#### 💭 个人思考
- **收获**: ...
- **疑问解答**: ...

#### 📝 相关学习材料
- 新增代码: `learning_materials/rope_multi_freq.py`
```

**Update Algorithm**:
1. Check if today's date section exists (format: `### YYYY-MM-DD: Topic`)
2. If exists → Append subsection under same date
3. If not exists → Insert new section maintaining chronological order
4. Extract: tasks (✅), problems (🐛), reflections (💭), materials (📝)

#### knowledge_base.md - Topical Knowledge Base
```markdown
**Q20: 为什么 RoPE 需要多频率？** ⭐️

A: 因为单一低频率受浮点数精度限制，无法区分相邻位置。

**详细说明**:
- 浮点数精度约为 10^-7
- 单一低频率 θ=10000 时，相邻位置差异 < 10^-7
- 使用多频率组合可以在不同尺度上编码位置信息

**代码示例**:
```python
# 验证浮点数精度限制
import torch
theta = 10000
pos_diff = 1 / theta  # 相邻位置差异
print(f"Position difference: {pos_diff}")  # 0.0001
```

参考代码: `learning_materials/rope_multi_freq.py`

---
```

**Update Algorithm**:
1. Scan existing Q numbers using regex `Q(\d+)`
2. Find max number (e.g., Q19) → new number = Q20
3. Infer topic category from content:
   - Keywords "归一化", "Norm" → 归一化技术
   - Keywords "位置", "RoPE", "编码" → 位置编码
   - Keywords "注意力", "Attention" → 注意力机制
   - Keywords "训练", "优化", "DPO", "PPO" → 训练技术
4. Insert at end of relevant section or in "问答记录" area
5. Mark important questions with ⭐️ (if contains: 原理, 为什么, 核心)

#### learning_materials/README.md - Code Index
```markdown
## 位置编码 (Position Encoding)

- **`rope_basics.py`** - RoPE 基础实现
  - 演示旋转位置编码的核心机制
  - 可视化二维旋转变换

- **`rope_multi_freq.py`** ⭐️ - 多频率机制验证
  - 验证浮点数精度限制
  - 对比单频率 vs 多频率效果
  - 演示频率分配策略
```

**Update Algorithm**:
1. Detect new .py file creation in `learning_materials/`
2. Extract description from file docstring or top comments
3. Categorize by topic (归一化/位置编码/注意力/前馈网络/训练技术)
4. Insert at end of category section
5. Mark foundational files with ⭐️

#### notes.md - Master Index
Only update when:
- New major section added to knowledge_base.md
- New date added to learning_log.md (update "按日期查找")
- File structure changes significantly

### Git Automation

**Commit Message Generation**:
```python
# Pattern: "学习 [主题] [子主题]" or "[动作] [对象]"

Examples:
- "学习 RMSNorm 归一化原理"
- "理解 RoPE 多频率机制"
- "添加 Attention 学习材料"
- "解决 CUDA 内存溢出问题"
- "完善位置编码知识点"

Algorithm:
1. Extract primary MiniMind term from content
2. Identify action type (学习/理解/添加/解决/完善)
3. Add sub-topic if present
4. Limit to 30 characters
5. Remove generic phrases like "Generated with Claude Code"
```

**Git Workflow**:
```bash
# Automatic sequence
cd {user_repo_root}
git add docs/notes.md docs/learning_log.md docs/knowledge_base.md docs/learning_materials/
git commit -m "{generated_message}"
git push origin {current_branch}

# Error handling
- Network timeout → Retry 3 times (exponential backoff: 1s, 2s, 4s)
- Push rejected → Log warning, suggest git pull --rebase
- Permission error → Log error, skip push
```

## MiniMind Terminology Database

### Architecture Components (20 terms)
```
RMSNorm, LayerNorm, BatchNorm, GroupNorm,
RoPE, YaRN, ALiBi, SinusoidalPE,
Attention, MultiHeadAttention, GQA, MQA, FlashAttention,
FeedForward, SwiGLU, GELU, GLU,
Transformer, TransformerBlock, CausalLM
```

### Training Methods (20 terms)
```
pretrain, pretraining,
SFT, supervised fine-tuning,
LoRA, LoRA-r, LoRA-alpha,
DPO, Direct Preference Optimization,
PPO, Proximal Policy Optimization,
GRPO, Group Relative Policy Optimization,
SPO, Simple Policy Optimization,
RLHF, RLAIF,
distillation, knowledge distillation,
teacher-student, white-box distillation
```

### Model Variants (10 terms)
```
MiniMind-Dense, MiniMind-MoE,
MiniMind-Reason, R1-style,
Mixture of Experts, MoE, shared experts, routed experts,
expert routing, load balancing loss
```

### Configuration (10 terms)
```
hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads,
vocab_size, max_seq_len, max_position_embeddings,
rope_theta, rope_scaling,
flash_attn
```

### Module Mapping

| Concept | Module Path |
|---------|-------------|
| RMSNorm, LayerNorm | modules/01-foundation/01-normalization/ |
| RoPE, YaRN | modules/01-foundation/02-position-encoding/ |
| Attention, GQA | modules/01-foundation/03-attention/ |
| FeedForward, SwiGLU | modules/01-foundation/04-feedforward/ |
| Transformer | modules/02-architecture/01-transformer-block/ |
| Training Pipeline | modules/02-architecture/02-complete-model/ |

## Usage

### Installation

1. Copy this skill to your Claude Code skills directory:
```bash
cp -r minimind-learning-skill ~/.claude/skills/
```

2. Ensure your MiniMind repository has the following structure:
```
your-minimind-fork/
├── model/                   # MiniMind source code
├── trainer/
├── docs/                    # Will be created by skill
│   ├── notes.md
│   ├── learning_log.md
│   ├── knowledge_base.md
│   └── learning_materials/
│       ├── README.md
│       └── *.py
└── ...
```

3. (Optional) Configure skill behavior by creating `.minimind-learning.json`:
```json
{
  "auto_commit": true,
  "auto_push": true,
  "batch_delay": 5,
  "git": {
    "remote": "origin",
    "branch": "master",
    "retry_count": 3,
    "timeout": 30
  }
}
```

### Quick Start

Just start chatting about MiniMind! The skill will automatically:

**Example 1: Learning New Concept**
```
You: 什么是 RMSNorm？
Claude: [Explains RMSNorm...]

# Behind the scenes:
# ✅ learning_log.md updated with today's entry
# ✅ knowledge_base.md gets new Q20
# ✅ Git committed: "学习 RMSNorm 归一化原理"
# ✅ Git pushed to origin
```

**Example 2: Solving Problem**
```
You: 运行训练时报错 CUDA out of memory，怎么办？
Claude: [Provides solution...]

# Behind the scenes:
# ✅ learning_log.md gets "遇到的问题" section
# ✅ Extracts: error phenomenon, root cause, solution
# ✅ Git committed: "解决 CUDA 内存溢出问题"
```

**Example 3: Explicit Request**
```
You: 我刚理解了 RoPE 的多频率机制，记录一下
Claude: [Updates notes...]

# Behind the scenes:
# ✅ All three files updated
# ✅ Git committed: "理解 RoPE 多频率机制"
```

### Configuration Options

Create `.minimind-learning.json` in your repository root:

```json
{
  "auto_commit": true,        // Auto commit after updates
  "auto_push": true,          // Auto push to remote
  "batch_delay": 5,           // Seconds to wait before batch update (Tier 2)
  "git": {
    "remote": "origin",       // Git remote name
    "branch": "master",       // Default branch
    "retry_count": 3,         // Push retry attempts
    "timeout": 30             // Git operation timeout (seconds)
  },
  "notes_dir": "docs",        // Notes directory (default: docs)
  "mark_important": true      // Auto mark important Q&A with ⭐️
}
```

## Skill Instructions

When this skill is activated, follow these instructions:

### Core Behavior

1. **Passive Monitoring**: Always monitor conversations for MiniMind-related content without announcing yourself
2. **Silent Updates**: Update notes in the background without asking for confirmation
3. **Smart Batching**: Group related updates together (Tier 2 triggers)
4. **Clean Git History**: Generate concise, meaningful commit messages

### Content Extraction

**From User Messages**:
```python
# Extract questions
patterns = [
    r"^(.*[?？])$",  # Question mark ending
    r"^(什么是|如何|为什么|怎样|解释|原理)(.*?)([?？。]|$)",
    r"^(.*)(吗|呢)[?？。]*$"
]

# Extract problems
problem_markers = ["报错", "错误", "失败", "不工作", "问题", "Bug"]
```

**From Claude Responses**:
```python
# Extract concepts and definitions
patterns = [
    r"([A-Z\u4e00-\u9fa5]+)\s*(是|：)(.*?)([。\n]|$)",  # "RMSNorm 是..."
    r"\*\*([^*]+)\*\*\s*[：:](.*?)([。\n]|$)",        # **概念**: 定义
    r"###\s+([^\n]+)\n\n([^\n]+)"                    # ### 标题
]

# Extract code examples
code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
```

### Update Decision Flow

```
Conversation → Trigger Detection
                ↓
         ┌──────┴──────┐
         │             │
    Tier 1/3        Tier 2
    (Immediate)   (Delayed 5s)
         │             │
         └──────┬──────┘
                ↓
        Content Extraction
                ↓
        ┌───────┼───────┐
        ↓       ↓       ↓
    log.md   kb.md   materials/
        └───────┬───────┘
                ↓
          Git Commit
                ↓
           Git Push
```

### File Update Strategies

**learning_log.md**:
```python
def update_learning_log(date, topic, tasks, problems, reflections, materials):
    # 1. Check if date section exists
    date_pattern = f"### {date}:"
    if date_pattern in content:
        # Append subsection
        insert_after(date_pattern, new_subsection)
    else:
        # Find correct position (chronological order)
        all_dates = extract_dates(content)
        insert_position = find_insert_position(all_dates, date)
        insert_section(insert_position, new_date_section)
```

**knowledge_base.md**:
```python
def update_knowledge_base(question, answer, code_example, category):
    # 1. Find next Q number
    existing_qs = re.findall(r"Q(\d+)", content)
    next_q = max(existing_qs) + 1 if existing_qs else 1

    # 2. Determine category
    category_mapping = {
        "归一化": ["归一化", "Norm", "RMS", "Layer"],
        "位置编码": ["位置", "RoPE", "YaRN", "编码"],
        "注意力": ["注意力", "Attention", "GQA", "MQA"],
        "前馈": ["FeedForward", "SwiGLU", "GLU"],
        "训练": ["训练", "DPO", "PPO", "LoRA", "SFT"]
    }
    inferred_category = infer_category(question, category_mapping)

    # 3. Insert at category end or in Q&A section
    insert_at_category_end(inferred_category, qa_entry)

    # 4. Mark important
    if any(keyword in question for keyword in ["原理", "为什么", "核心", "本质"]):
        mark_as_important(qa_entry)
```

**learning_materials/README.md**:
```python
def update_materials_readme(new_file_path):
    # 1. Extract metadata from file
    docstring = extract_docstring(new_file_path)
    category = infer_category_from_filename(new_file_path)

    # 2. Generate entry
    entry = f"- **`{filename}`** - {description}\n"
    entry += format_bullet_points(docstring_lines)

    # 3. Insert at category end
    insert_at_category_end(category, entry)

    # 4. Mark foundational files
    if is_foundational(filename):  # e.g., "basics", "explained"
        mark_with_star(entry)
```

### Git Message Generation

```python
def generate_commit_message(changes):
    # 1. Identify primary action
    actions = {
        "learning_log": "学习",
        "problem_solving": "解决",
        "code_creation": "添加",
        "concept_clarification": "理解",
        "refactoring": "完善"
    }

    # 2. Extract primary MiniMind term
    terms_found = []
    for term in MINIMIND_TERMS:
        if term.lower() in changes.content.lower():
            terms_found.append(term)
    primary_term = terms_found[0] if terms_found else "知识点"

    # 3. Extract sub-topic
    sub_topic = extract_sub_topic(changes.content)  # e.g., "多频率机制"

    # 4. Construct message
    action = actions[changes.type]
    message = f"{action} {primary_term}"
    if sub_topic:
        message += f" {sub_topic}"

    # 5. Limit length
    return message[:30]

# Examples:
# "学习 RMSNorm 归一化原理"
# "解决 CUDA 内存溢出问题"
# "添加 RoPE 学习材料"
# "理解 Attention 计算流程"
```

### Error Handling

**File Not Found**:
```python
def ensure_notes_structure():
    notes_dir = "docs"
    files = {
        "notes.md": NOTES_TEMPLATE,
        "learning_log.md": LOG_TEMPLATE,
        "knowledge_base.md": KB_TEMPLATE,
        "learning_materials/README.md": MATERIALS_TEMPLATE
    }

    for file, template in files.items():
        filepath = os.path.join(notes_dir, file)
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(template)
```

**Git Push Failures**:
```python
def safe_git_push(max_retries=3):
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["git", "push", "origin", branch],
                timeout=30,
                capture_output=True
            )
            if result.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)

    # Log failure but don't block
    log_warning("Git push failed after 3 attempts. Changes committed locally.")
    return False
```

**Concurrent Updates**:
```python
from filelock import FileLock

def update_file_safely(filepath, update_func):
    lock_path = f"{filepath}.lock"
    with FileLock(lock_path, timeout=10):
        content = read_file(filepath)
        new_content = update_func(content)
        write_file(filepath, new_content)
```

### Working Directory Detection

```python
def detect_user_repo():
    # Find git repository root
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True
    )
    repo_root = result.stdout.strip()

    # Verify it's a MiniMind repo
    indicators = [
        "model/model_minimind.py",
        "trainer/train_pretrain.py",
        "README.md"  # Contains "MiniMind"
    ]

    if all(os.path.exists(os.path.join(repo_root, ind)) for ind in indicators):
        return repo_root
    else:
        raise ValueError("Not a valid MiniMind repository")

# Create docs/ in user's repo
docs_dir = os.path.join(detect_user_repo(), "docs")
os.makedirs(docs_dir, exist_ok=True)
```

## Templates

Templates are stored in `templates/` directory and used to initialize missing files.

### Validation

Run validation script to check note consistency:
```bash
cd minimind-learning-skill
python scripts/validate_notes.py
```

Checks:
- Q numbers are sequential (Q1, Q2, Q3, ...)
- Date format is consistent (YYYY-MM-DD)
- No orphaned references (all mentioned files exist)
- Git commit messages follow convention

## Troubleshooting

**Problem**: Notes not updating
- **Check**: Is this a MiniMind-related conversation?
- **Check**: Are trigger keywords present? (See Tier 1 list)
- **Solution**: Use explicit request: "记录这个知识点"

**Problem**: Git push fails
- **Check**: Network connection
- **Check**: Git credentials configured
- **Solution**: Changes are committed locally, manually push later

**Problem**: Q numbers skip (Q1, Q2, Q5...)
- **Check**: Manual edits to knowledge_base.md?
- **Solution**: Run validation script to fix numbering

**Problem**: Duplicate entries
- **Check**: Same concept discussed multiple times
- **Solution**: Merge duplicate Q&A manually, skill will avoid duplicates in future

## Contributing

This skill is designed for the MiniMind learning community. Contributions welcome!

**How to contribute**:
1. Fork the repository
2. Create feature branch
3. Test with real learning scenarios
4. Submit pull request

**Areas for improvement**:
- Add support for other languages (English, Japanese)
- Integrate with Anki/Obsidian for spaced repetition
- Support voice input for notes
- Generate visual diagrams from concepts

## License

MIT License - Free to use and modify for educational purposes.

## Credits

- **Author**: Joye Huang (joyehuang)
- **Inspired by**: MiniMind project by jingyaogong
- **Community**: MiniMind learning group members

---

**Version History**:
- v1.0.0 (2026-02-23): Initial release
  - Three-tier triggering system
  - Full Git automation
  - 50+ MiniMind term recognition
  - Three-file note system

---

*This skill is part of the MiniMind educational ecosystem. For more information, visit [MiniMind GitHub](https://github.com/jingyaogong/minimind).*
