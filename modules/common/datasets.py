"""
⚠️ 此文件已废弃

datasets.py 已重命名为 data_sources.py 以避免与 HuggingFace datasets 库的命名冲突。

迁移方法：
    旧代码:
        from modules.common.datasets import get_experiment_data

    新代码:
        from modules.common.data_sources import get_experiment_data

详见: https://github.com/joyehuang/minimind-notes/pull/20
"""

# 抛出清晰的错误信息
raise ImportError(
    "\n\n"
    "=" * 70 + "\n"
    "⚠️  datasets.py 已废弃\n"
    "=" * 70 + "\n\n"
    "此文件已重命名为 data_sources.py 以避免与 HuggingFace datasets 库冲突。\n\n"
    "请更新你的导入语句：\n\n"
    "  旧: from modules.common.datasets import get_experiment_data\n"
    "  新: from modules.common.data_sources import get_experiment_data\n\n"
    "详见: https://github.com/joyehuang/minimind-notes/pull/20\n"
    "=" * 70 + "\n"
)
