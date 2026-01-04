## 📋 PR 说明

这个 PR 包含了将 MiniMind 学习笔记迁移到 VitePress 的详细方案和计划文档。

## 📦 包含的文档

| 文档 | 说明 |
|------|------|
| **VITEPRESS_MIGRATION_PLAN.md** ⭐ | 详细的迁移计划，包含 5 个阶段的任务清单 |
| **VITEPRESS_RECOMMENDED_STRUCTURE.md** | 针对本项目的推荐方案 (方案 A+) |
| **VITEPRESS_SETUP_GUIDE.md** | VitePress 完整实施指南 |
| **.vitepress-config-example.ts** | VitePress 配置示例 |
| **docs-index-example.md** | 首页设计示例 |

## 🎯 方案选择: 方案 A+ (混合方案)

### 核心原则

- ✅ **所有现有文件保持在原位置**
  - `learning_log.md`, `knowledge_base.md` 不移动
  - `modules/`, `learning_materials/` 不移动
  - Claude Code 工作流不受影响

- ✅ **VitePress 作为"展示层"**
  - 通过 `srcDir: '.'` 读取根目录
  - 直接访问所有现有 .md 文件
  - 无需符号链接或文件复制

- ✅ **快速部署到 Vercel**
  - 3-4 小时完成基础迁移 (不含图片)
  - 零配置或简单配置
  - 每次 push 自动部署

- ✅ **不破坏现有工作流**
  - CLAUDE.md 的路径指令继续有效
  - Python 脚本可以直接运行
  - Git 历史完整保留

## 📐 目录结构设计

```
minimind-notes/
├── .vitepress/                    # 新增: VitePress 配置
│   ├── config.ts
│   └── theme/
├── docs/                          # 新增: 新内容目录
│   ├── index.md                   # 首页
│   ├── guide/                     # 学习指南
│   └── public/                    # 静态资源
│
├── learning_log.md                # ✅ 保持原位
├── knowledge_base.md              # ✅ 保持原位
├── notes.md                       # ✅ 保持原位
├── CLAUDE.md                      # ✅ 保持原位
├── modules/                       # ✅ 保持原位
├── learning_materials/            # ✅ 保持原位
└── [其他文件保持不变]
```

## 📊 迁移计划 (5个阶段)

### Phase 1: 基础设置 ⏱️ 30分钟
- 安装 VitePress 和依赖
- 创建 `.vitepress/config.ts` 基础配置
- 配置 `package.json` 脚本
- 测试本地运行

### Phase 2: 首页和导航 ⏱️ 1小时
- 创建 `docs/index.md` 首页
- 配置顶部导航和侧边栏
- 创建学习指南页面
- 测试所有链接

### Phase 3: 内容优化 ⏱️ 1-2小时
- 为关键文件添加 frontmatter
- 优化 markdown 格式
- 创建自定义样式
- 配置搜索功能

### Phase 4: 可视化资源 ⏱️ 后续补充
- 运行 Python 脚本生成图表
- 组织图片到 `docs/public/images/`
- 在文档中引用图片
- ⚠️ **可后置，不阻塞部署**

### Phase 5: 部署 ⏱️ 30分钟
- 配置 Vercel 部署
- 测试部署
- (可选) 配置 GitHub Pages

**总计**: 3-4小时 (不含 Phase 4)

## 🖼️ 关于可视化图片处理

### 建议策略
1. **先搭建框架** - Phase 1-3 完成后即可部署
2. **后续补充图片** - 本地运行脚本，保存到 `docs/public/images/`
3. **逐步完善** - 每个模块逐步添加可视化

### 展示方式
```markdown
## 示例: RoPE 可视化

::: details 查看源代码
```python
# learning_materials/rope_basics.py
代码...
\```
:::

![RoPE 输出](/images/visualizations/rope_basics.png)

### 🚀 本地运行
\`\`\`bash
python learning_materials/rope_basics.py
\`\`\`
```

## 🚨 风险控制

| 风险 | 预防措施 | 应对方案 |
|------|---------|---------|
| 链接失效 | 使用相对路径，充分测试 | 检查 VitePress 链接规则 |
| 构建失败 | 本地充分测试 | 检查构建日志 |
| 工作流被破坏 | 不移动文件，保持路径 | 立即回滚 |
| 样式混乱 | 先用默认主题 | 浏览器调试 |

## ✅ Review Checklist

请 review 以下方面:

- [ ] **方案合理性**
  - [ ] 方案 A+ 适合本项目吗?
  - [ ] 有没有更好的替代方案?

- [ ] **计划可行性**
  - [ ] 5 个阶段的划分合理吗?
  - [ ] 时间估算准确吗?
  - [ ] 任务清单完整吗?

- [ ] **风险可控性**
  - [ ] 是否会破坏现有工作流?
  - [ ] 回滚机制是否充分?
  - [ ] 有没有遗漏的风险?

- [ ] **图片处理策略**
  - [ ] 后置图片生成合理吗?
  - [ ] 展示方式是否满足需求?

- [ ] **部署方案**
  - [ ] Vercel 部署配置是否正确?
  - [ ] 需要 GitHub Pages 备份吗?

## 🚀 下一步

Review 通过后:
1. ✅ 合并这个 PR
2. ✅ 创建新分支 `feature/vitepress-migration`
3. ✅ 按照 `VITEPRESS_MIGRATION_PLAN.md` 逐步执行
4. ✅ 每完成一个 Phase 提交一次

## 💬 问题和建议

如果有任何问题或建议，请在 PR 中评论，我会及时调整方案。

---

**相关文档**:
- 📋 [完整迁移计划](./VITEPRESS_MIGRATION_PLAN.md)
- 🏗️ [推荐结构说明](./VITEPRESS_RECOMMENDED_STRUCTURE.md)
- 📖 [实施指南](./VITEPRESS_SETUP_GUIDE.md)
