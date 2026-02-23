@echo off
REM init-for-learning.bat - Windows 版本初始化脚本
setlocal enabledelayedexpansion

echo.
echo 🚀 MiniMind Learning Skill 初始化
echo ==================================
echo.

REM 检查是否在正确的目录
if not exist "minimind-learning-skill" (
    echo ❌ 错误：未找到 minimind-learning-skill\ 目录
    echo    请确保在 minimind-notes 仓库根目录运行此脚本
    pause
    exit /b 1
)

REM 1. 备份示例笔记
echo 📦 第 1 步：备份作者的学习记录...
if exist "learning_log.md" (
    if not exist "example-notes" mkdir example-notes

    if exist "learning_log.md" move /y learning_log.md example-notes\ >nul
    if exist "knowledge_base.md" move /y knowledge_base.md example-notes\ >nul
    if exist "notes.md" move /y notes.md example-notes\ >nul
    if exist "learning_materials" move /y learning_materials example-notes\ >nul
    if exist "NOTE_UPDATE_GUIDE.md" move /y NOTE_UPDATE_GUIDE.md example-notes\ >nul

    echo    ✅ 示例笔记已保存到 example-notes\ 目录
    echo    💡 你可以随时查看作为参考
) else (
    echo    ℹ️  未找到示例笔记（可能已经清理过）
)

echo.

REM 2. 安装 skill
echo 📥 第 2 步：安装 MiniMind Learning Skill...

set "SKILL_NAME=minimind-learning"
set "CLAUDE_DIR=%USERPROFILE%\.claude"
set "SKILLS_DIR=%CLAUDE_DIR%\skills"
set "SKILL_TARGET=%SKILLS_DIR%\%SKILL_NAME%"

REM 创建 skills 目录
if not exist "%SKILLS_DIR%" mkdir "%SKILLS_DIR%"

REM 复制 skill
if exist "%SKILL_TARGET%" (
    echo    ⚠️  检测到已安装的 skill
    set /p "REPLY=   是否覆盖？[y/N] "
    if /i "!REPLY!"=="y" (
        rmdir /s /q "%SKILL_TARGET%"
        xcopy /s /e /i /q minimind-learning-skill "%SKILL_TARGET%" >nul
        echo    ✅ Skill 已更新到 %SKILL_TARGET%
    ) else (
        echo    ⏭️  跳过 skill 安装
    )
) else (
    xcopy /s /e /i /q minimind-learning-skill "%SKILL_TARGET%" >nul
    echo    ✅ Skill 已安装到 %SKILL_TARGET%
)

echo.

REM 3. 配置 .gitignore
echo 📝 第 3 步：配置 Git...
findstr /c:"docs/" .gitignore >nul 2>&1
if errorlevel 1 (
    echo docs/>> .gitignore
    echo    ✅ 已添加 docs/ 到 .gitignore
) else (
    echo    ℹ️  .gitignore 已配置
)

echo.

REM 4. 完成
echo ✅ 初始化完成！
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 📚 下一步：开始学习
echo.
echo 在你的 AI Coding Agent 中（任选其一）：
echo.
echo   • Claude Code (CLI):
echo     $ claude code
echo.
echo   • VS Code + Claude 扩展:
echo     打开 VS Code，启动 Claude
echo.
echo   • Cursor / Windsurf / 其他 IDE:
echo     启动 AI 助手
echo.
echo 然后说：'开始学习' 或 '开始今天的学习'
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 💡 提示：
echo   • 你的学习笔记将保存在 docs\ 目录
echo   • 作者的示例笔记在 example-notes\ 可供参考
echo   • 详细使用说明：type USER_GUIDE.md
echo.
pause
