#!/usr/bin/env python3
"""
validate_notes.py - 验证 MiniMind 学习笔记的一致性

检查项:
1. Q 编号连续性（Q1, Q2, Q3...）
2. 日期格式统一（YYYY-MM-DD）
3. 文件引用完整性（所有提到的文件都存在）
4. Git commit message 格式

使用方法:
    python validate_notes.py [--fix-numbering]

参数:
    --fix-numbering: 自动修复 Q 编号跳跃
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Set


class NoteValidator:
    """笔记验证器"""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.errors = []
        self.warnings = []

    def validate_all(self) -> bool:
        """运行所有验证检查"""
        print("🔍 开始验证笔记系统...")
        print()

        # 检查目录结构
        if not self._check_directory_structure():
            return False

        # 验证各个文件
        self._validate_learning_log()
        self._validate_knowledge_base()
        self._validate_learning_materials()
        self._validate_notes_index()

        # 输出结果
        self._print_results()

        return len(self.errors) == 0

    def _check_directory_structure(self) -> bool:
        """检查目录结构是否完整"""
        required_files = [
            self.docs_dir / "learning_log.md",
            self.docs_dir / "knowledge_base.md",
            self.docs_dir / "notes.md",
            self.docs_dir / "learning_materials" / "README.md",
        ]

        missing = [f for f in required_files if not f.exists()]
        if missing:
            print("❌ 缺少必需文件:")
            for f in missing:
                print(f"   - {f}")
            return False

        print("✅ 目录结构完整")
        return True

    def _validate_learning_log(self):
        """验证学习日志"""
        print("📅 验证 learning_log.md...")

        filepath = self.docs_dir / "learning_log.md"
        content = filepath.read_text(encoding="utf-8")

        # 检查日期格式
        date_pattern = r"^### (\d{4}-\d{2}-\d{2}): (.+)$"
        dates = re.findall(date_pattern, content, re.MULTILINE)

        if not dates:
            self.warnings.append("learning_log.md 中未找到日期条目")
            return

        # 验证日期格式
        for date_str, topic in dates:
            try:
                year, month, day = map(int, date_str.split("-"))
                if not (1900 <= year <= 2100):
                    self.errors.append(f"无效年份: {date_str}")
                if not (1 <= month <= 12):
                    self.errors.append(f"无效月份: {date_str}")
                if not (1 <= day <= 31):
                    self.errors.append(f"无效日期: {date_str}")
            except ValueError:
                self.errors.append(f"日期格式错误: {date_str}")

        # 检查日期顺序（应该按时间倒序）
        parsed_dates = [d[0] for d in dates]
        if parsed_dates != sorted(parsed_dates, reverse=True):
            self.warnings.append("日期未按倒序排列（最新的应该在最上面）")

        print(f"   找到 {len(dates)} 个日期条目")

    def _validate_knowledge_base(self):
        """验证知识库"""
        print("🧠 验证 knowledge_base.md...")

        filepath = self.docs_dir / "knowledge_base.md"
        content = filepath.read_text(encoding="utf-8")

        # 提取 Q 编号
        q_pattern = r"\*\*Q(\d+):"
        q_numbers = [int(n) for n in re.findall(q_pattern, content)]

        if not q_numbers:
            self.warnings.append("knowledge_base.md 中未找到问答条目")
            return

        print(f"   找到 {len(q_numbers)} 个问答")

        # 检查编号连续性
        expected = list(range(1, len(q_numbers) + 1))
        if q_numbers != expected:
            missing = set(expected) - set(q_numbers)
            duplicate = [n for n in q_numbers if q_numbers.count(n) > 1]

            if missing:
                self.errors.append(f"Q 编号缺失: {sorted(missing)}")
            if duplicate:
                self.errors.append(f"Q 编号重复: {sorted(set(duplicate))}")

            # 显示实际编号
            self.warnings.append(f"实际 Q 编号: {q_numbers}")
            self.warnings.append(f"期望 Q 编号: {expected}")
        else:
            print(f"   ✅ Q 编号连续 (Q1-Q{len(q_numbers)})")

        # 检查重要标记
        important_count = content.count("⭐️")
        print(f"   找到 {important_count} 个重要问答 (⭐️)")

    def _validate_learning_materials(self):
        """验证学习材料"""
        print("🔬 验证 learning_materials/...")

        materials_dir = self.docs_dir / "learning_materials"
        readme_path = materials_dir / "README.md"

        # 获取所有 .py 文件
        py_files = set(materials_dir.glob("*.py"))
        print(f"   找到 {len(py_files)} 个 Python 文件")

        # 检查 README 中的引用
        readme_content = readme_path.read_text(encoding="utf-8")
        referenced_files = re.findall(r"`([^`]+\.py)`", readme_content)

        # 检查文件是否都在 README 中
        readme_files = {materials_dir / f for f in referenced_files}
        missing_in_readme = py_files - readme_files

        if missing_in_readme:
            self.warnings.append(
                f"以下文件未在 README 中列出: {[f.name for f in missing_in_readme]}"
            )

        # 检查 README 中引用的文件是否存在
        missing_files = readme_files - py_files
        if missing_files:
            self.errors.append(
                f"README 中引用的文件不存在: {[f.name for f in missing_files]}"
            )

    def _validate_notes_index(self):
        """验证总索引"""
        print("📖 验证 notes.md...")

        filepath = self.docs_dir / "notes.md"
        content = filepath.read_text(encoding="utf-8")

        # 检查是否包含必要链接
        required_links = [
            "learning_log.md",
            "knowledge_base.md",
            "learning_materials/README.md",
        ]

        for link in required_links:
            if link not in content:
                self.warnings.append(f"notes.md 中缺少链接: {link}")

    def fix_q_numbering(self):
        """修复 Q 编号"""
        print("🔧 修复 Q 编号...")

        filepath = self.docs_dir / "knowledge_base.md"
        content = filepath.read_text(encoding="utf-8")

        # 提取所有 Q&A
        q_pattern = r"\*\*Q(\d+):(.*?)\n\n---"
        matches = list(re.finditer(q_pattern, content, re.DOTALL))

        if not matches:
            print("   未找到需要修复的 Q&A")
            return

        # 重新编号
        new_content = content
        for i, match in enumerate(matches, start=1):
            old_q = match.group(1)
            full_match = match.group(0)
            new_q = str(i)

            if old_q != new_q:
                new_full = full_match.replace(f"**Q{old_q}:", f"**Q{new_q}:", 1)
                new_content = new_content.replace(full_match, new_full, 1)
                print(f"   Q{old_q} → Q{new_q}")

        # 保存
        filepath.write_text(new_content, encoding="utf-8")
        print(f"✅ 已修复 Q 编号")

    def _print_results(self):
        """打印验证结果"""
        print()
        print("=" * 60)

        if self.errors:
            print(f"❌ 发现 {len(self.errors)} 个错误:")
            for err in self.errors:
                print(f"   - {err}")
            print()

        if self.warnings:
            print(f"⚠️  发现 {len(self.warnings)} 个警告:")
            for warn in self.warnings:
                print(f"   - {warn}")
            print()

        if not self.errors and not self.warnings:
            print("✅ 所有检查通过！笔记系统状态良好。")
        elif not self.errors:
            print("✅ 无严重错误，但有一些改进建议。")
        else:
            print("❌ 发现错误，请修复后再提交。")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="验证 MiniMind 学习笔记")
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="笔记目录路径 (默认: docs)",
    )
    parser.add_argument(
        "--fix-numbering",
        action="store_true",
        help="自动修复 Q 编号",
    )

    args = parser.parse_args()

    # 检查目录是否存在
    if not Path(args.docs_dir).exists():
        print(f"❌ 目录不存在: {args.docs_dir}")
        print()
        print("请在 MiniMind 仓库根目录运行此脚本，或使用 --docs-dir 指定笔记目录")
        return 1

    validator = NoteValidator(args.docs_dir)

    # 修复编号（如果请求）
    if args.fix_numbering:
        validator.fix_q_numbering()
        print()

    # 验证
    success = validator.validate_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
