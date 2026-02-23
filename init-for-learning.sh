#!/bin/bash
# init-for-learning.sh - 初始化 MiniMind 学习环境

set -e

echo "🚀 MiniMind Learning Skill 初始化"
echo "=================================="
echo ""

# 检查是否在正确的目录
if [ ! -d "minimind-learning-skill" ]; then
    echo "❌ 错误：未找到 minimind-learning-skill/ 目录"
    echo "   请确保在 minimind-notes 仓库根目录运行此脚本"
    exit 1
fi

# 1. 备份示例笔记
echo "📦 第 1 步：备份作者的学习记录..."
if [ -f "learning_log.md" ] || [ -f "knowledge_base.md" ]; then
    mkdir -p example-notes

    [ -f "learning_log.md" ] && mv learning_log.md example-notes/
    [ -f "knowledge_base.md" ] && mv knowledge_base.md example-notes/
    [ -f "notes.md" ] && mv notes.md example-notes/
    [ -d "learning_materials" ] && mv learning_materials example-notes/
    [ -f "NOTE_UPDATE_GUIDE.md" ] && mv NOTE_UPDATE_GUIDE.md example-notes/

    echo "   ✅ 示例笔记已保存到 example-notes/ 目录"
    echo "   💡 你可以随时查看作为参考"
else
    echo "   ℹ️  未找到示例笔记（可能已经清理过）"
fi

echo ""

# 2. 安装 skill
echo "📥 第 2 步：安装 MiniMind Learning Skill..."

# 检测 Claude skills 目录
SKILL_NAME="minimind-learning"
CLAUDE_DIR="$HOME/.claude"
SKILLS_DIR="$CLAUDE_DIR/skills"

# 创建 skills 目录（如果不存在）
mkdir -p "$SKILLS_DIR"

# 复制 skill
SKILL_TARGET="$SKILLS_DIR/$SKILL_NAME"
if [ -d "$SKILL_TARGET" ]; then
    echo "   ⚠️  检测到已安装的 skill"
    read -p "   是否覆盖？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   ⏭️  跳过 skill 安装"
    else
        rm -rf "$SKILL_TARGET"
        cp -r minimind-learning-skill "$SKILL_TARGET"
        echo "   ✅ Skill 已更新到 $SKILL_TARGET"
    fi
else
    cp -r minimind-learning-skill "$SKILL_TARGET"
    echo "   ✅ Skill 已安装到 $SKILL_TARGET"
fi

echo ""

# 3. 创建 .gitignore（忽略用户笔记）
echo "📝 第 3 步：配置 Git..."
if ! grep -q "^docs/" .gitignore 2>/dev/null; then
    echo "docs/" >> .gitignore
    echo "   ✅ 已添加 docs/ 到 .gitignore"
else
    echo "   ℹ️  .gitignore 已配置"
fi

echo ""

# 4. 完成
echo "✅ 初始化完成！"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 下一步：开始学习"
echo ""
echo "在你的 AI Coding Agent 中（任选其一）："
echo ""
echo "  • Claude Code (CLI):"
echo "    $ claude code"
echo ""
echo "  • VS Code + Claude 扩展:"
echo "    打开 VS Code，启动 Claude"
echo ""
echo "  • Cursor / Windsurf / 其他 IDE:"
echo "    启动 AI 助手"
echo ""
echo "然后说：'开始学习' 或 '开始今天的学习'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 提示："
echo "  • 你的学习笔记将保存在 docs/ 目录"
echo "  • 作者的示例笔记在 example-notes/ 可供参考"
echo "  • 详细使用说明：cat USER_GUIDE.md"
echo ""
