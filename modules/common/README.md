# 通用工具 (Common Utilities)

本目录包含所有实验模块共享的工具代码。

## 📋 系统要求

### Python 版本
- **要求**: Python 3.10+

**注意**: 代码使用了 Python 3.10+ 的类型联合语法（`str | list`），低于此版本将无法运行。

### 依赖库
- `torch` - PyTorch 深度学习框架
- `requests` - HTTP 请求（用于数据下载）
- `datasets` - HuggingFace datasets 库（用于 TinyStories 下载）

安装方法：
```bash
pip install torch requests datasets
```

## 📦 可用工具

### data_sources.py - 数据集管理

提供统一的实验数据接口，支持：
- TinyShakespeare（经典字符级数据，1MB）
- TinyStories（现代英文，支持取子集）
- 合成数据（用于可视化实验）

**使用示例**:
```python
from modules.common.data_sources import get_experiment_data

# 获取 TinyShakespeare
text = get_experiment_data('shakespeare')

# 获取 TinyStories 子集（10MB）
texts = get_experiment_data('tinystories', size_mb=10)

# 生成合成数据
text = get_experiment_data('synthetic', size_mb=1)
```

**命令行使用**:
```bash
cd modules/common

# 下载所有数据集
python data_sources.py --download-all

# 测试单个数据集
python data_sources.py --dataset shakespeare
```

### experiment_base.py - 实验基类

提供统一的实验框架，包括：
- 自动设备检测（CPU/MPS/CUDA）
- 结果保存（图表 + 指标）
- 进度显示
- 可复现性（固定随机种子）

**使用示例**:
```python
from modules.common.experiment_base import Experiment

class MyExperiment(Experiment):
    def __init__(self):
        super().__init__(
            name="my_experiment",
            output_dir="experiments/results"
        )

    def run(self):
        # 你的实验代码
        metrics = {'accuracy': 0.95}
        self.print_metrics(metrics)
        self.save_metrics(metrics)

exp = MyExperiment()
exp.run()
```

### visualization.py - 可视化工具

提供常用的可视化函数。

**注意**: 此文件目前尚未创建，计划在后续模块中添加。

## ⚠️ 重要变更说明

### datasets.py 已重命名为 data_sources.py (2026-02)

**原因**: 避免与 HuggingFace `datasets` 库的命名冲突，该冲突会导致 TinyStories 数据集下载失败。

**背景**: Python 模块搜索时优先查找当前目录，如果存在本地 `datasets.py`，会导致 `from datasets import load_dataset` 错误导入本地文件而非 HuggingFace 库。

**迁移方法**:

| 旧代码 | 新代码 |
|--------|--------|
| `from modules.common.datasets import ...` | `from modules.common.data_sources import ...` |
| `python datasets.py --download-all` | `python data_sources.py --download-all` |

**注意**:
- `datasets.py` 文件已完全删除（不再存在于仓库中）
- 使用旧导入方式会收到标准的 `ModuleNotFoundError`
- 所有官方文档和实验代码已更新为新文件名
- Git 历史中仍可通过 `git log -- modules/common/datasets.py` 追溯旧文件

**相关信息**:
- 问题追踪: GitHub Issue #19
- 详细讨论: GitHub Pull Request #20

## 📝 贡献指南

在添加新工具时，请：
1. 在本 README 中添加工具说明
2. 在文件头部添加清晰的文档字符串
3. 提供使用示例
4. 确保工具是通用的，可以被多个模块复用
