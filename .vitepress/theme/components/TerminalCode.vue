<template>
  <section class="quick-start-section">
    <div class="start-container">
      <div class="start-header">
        <h2>{{ copy.title }}</h2>
        <p>{{ copy.subtitle }}</p>
      </div>

      <div class="terminal-wrapper">
        <div class="terminal-header">
          <div class="terminal-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
          </div>
          <div class="terminal-title">
            <span class="terminal-icon">⌘</span>
            <span>Terminal</span>
          </div>
        </div>

        <div class="terminal-content">
          <div class="code-block">
            <p class="comment">{{ copy.comments.clone }}</p>
            <p class="command">git clone https://github.com/joyehuang/minimind-notes.git</p>
            <p class="command">cd minimind-notes</p>
          </div>

          <div class="code-block">
            <p class="comment">{{ copy.comments.activate }}</p>
            <p class="command">source venv/bin/activate</p>
          </div>

          <div class="code-block">
            <p class="comment">{{ copy.comments.exp1 }}</p>
            <p class="command">cd modules/01-foundation/01-normalization/experiments</p>
            <p class="command success">python exp1_gradient_vanishing.py</p>
          </div>

          <div class="code-block">
            <p class="comment">{{ copy.comments.exp2 }}</p>
            <p class="command">cd ../../02-position-encoding/experiments</p>
            <p class="command success">python exp1_rope_basics.py</p>
          </div>

          <div class="code-block">
            <p class="comment">{{ copy.comments.exp3 }}</p>
            <p class="command">cd ../../03-attention/experiments</p>
            <p class="command success">python exp1_attention_basics.py</p>
          </div>
        </div>
      </div>

      <div class="outcomes-grid">
        <div class="outcome-card">
          <div class="outcome-icon">📊</div>
          <h4>{{ copy.outcomes.gradientTitle }}</h4>
          <p>{{ copy.outcomes.gradientDesc }}</p>
        </div>
        <div class="outcome-card">
          <div class="outcome-icon">🔄</div>
          <h4>{{ copy.outcomes.ropeTitle }}</h4>
          <p>{{ copy.outcomes.ropeDesc }}</p>
        </div>
        <div class="outcome-card">
          <div class="outcome-icon">🎯</div>
          <h4>{{ copy.outcomes.attnTitle }}</h4>
          <p>{{ copy.outcomes.attnDesc }}</p>
        </div>
      </div>

      <div class="start-footer">
        <a :href="withLocale('/ROADMAP')" class="start-link">
          {{ copy.footer }}
        </a>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useLocale } from '../i18n'

const { isEn, localePath } = useLocale()
const withLocale = (path: string) => `${localePath.value}${path}`

const copy = computed(() =>
  isEn.value
    ? {
        title: 'Quick Start',
        subtitle: '30 minutes, three experiments — change how you understand LLM training',
        comments: {
          clone: '# 1. Clone the repo',
          activate: '# 2. Activate your virtual environment (if any)',
          exp1: '# 3. Experiment 1: Why normalization?',
          exp2: '# 4. Experiment 2: Why RoPE position encoding?',
          exp3: '# 5. Experiment 3: How Attention works?'
        },
        outcomes: {
          gradientTitle: 'Gradient Vanishing',
          gradientDesc: 'Visualize gradient flow in deep networks',
          ropeTitle: 'RoPE Encoding',
          ropeDesc: 'See the math behind rotary position embeddings',
          attnTitle: 'Attention',
          attnDesc: 'Visualize attention weight computation'
        },
        footer: 'View the full learning roadmap →'
      }
    : {
        title: '快速开始',
        subtitle: '只需 30 分钟，三个实验，彻底改变你对 LLM 训练的理解',
        comments: {
          clone: '# 1. 克隆仓库',
          activate: '# 2. 激活虚拟环境（如果已有）',
          exp1: '# 3. 实验1：为什么需要归一化？',
          exp2: '# 4. 实验2：为什么用 RoPE 位置编码？',
          exp3: '# 5. 实验3：Attention 如何工作？'
        },
        outcomes: {
          gradientTitle: '梯度消失',
          gradientDesc: '可视化深层网络的梯度流动问题',
          ropeTitle: 'RoPE 编码',
          ropeDesc: '旋转位置编码的数学原理演示',
          attnTitle: 'Attention',
          attnDesc: '注意力权重的计算过程可视化'
        },
        footer: '查看完整学习路线 →'
      }
)
</script>

<style scoped>
.quick-start-section {
  padding: 3rem 0;
  border-bottom: 1px solid var(--vp-c-divider);
}

/* 暗夜主题优化 - shadcn 风格 */
:global(.dark) .quick-start-section {
  border-bottom: 1px solid var(--vp-c-divider);
}

.start-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

.start-header {
  text-align: center;
  margin-bottom: 3rem;
}

.start-header h2 {
  font-size: 2.2em;
  font-weight: 700;
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-1);
}

.start-header p {
  font-size: 1.1em;
  color: var(--vp-c-text-2);
  margin: 0;
}

.terminal-wrapper {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 2rem;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

.terminal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
}

.terminal-dots {
  display: flex;
  gap: 0.4rem;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.dot-red {
  background: #ef4444;
  opacity: 0.8;
}

.dot-yellow {
  background: #f59e0b;
  opacity: 0.8;
}

.dot-green {
  background: #10b981;
  opacity: 0.8;
}

.terminal-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.terminal-icon {
  font-size: 1.1em;
}

.terminal-content {
  padding: 1.5rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 0.9em;
  line-height: 1.8;
  background: #1e1e1e;
  color: #d4d4d4;
}

.code-block {
  margin-bottom: 1rem;
}

.code-block:last-child {
  margin-bottom: 0;
}

.comment {
  color: #6a9955;
  margin: 0 0 0.25rem 0;
}

.command {
  color: #d4d4d4;
  margin: 0;
}

.command.success {
  color: #4ec9b0;
}

.outcomes-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}

.outcome-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 1.25rem;
  text-align: center;
  transition: all 0.3s ease;
}

.outcome-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}

/* 暗夜主题卡片优化 - shadcn 风格 */
:global(.dark) .terminal-wrapper {
  background: #09090b;
  border: 1px solid var(--vp-c-divider);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

:global(.dark) .terminal-header {
  background: var(--vp-c-accent-bg);
  border-bottom: 1px solid var(--vp-c-divider);
}

:global(.dark) .terminal-title {
  color: var(--vp-c-text-2);
}

:global(.dark) .outcome-card {
  background: #18181b;
  border: 1px solid var(--vp-c-divider);
}

:global(.dark) .outcome-card:hover {
  background: var(--vp-c-accent-bg);
  border-color: var(--vp-c-text-2);
}

:global(.dark) .outcome-card h4 {
  color: var(--vp-c-text-1);
}

:global(.dark) .outcome-card p {
  color: var(--vp-c-text-2);
}

:global(.dark) .start-link {
  background: var(--vp-c-brand-1);
  color: #09090b;
}

:global(.dark) .start-link:hover {
  background: var(--vp-c-brand-2);
  color: #09090b;
}

:global(.dark) .start-header h2 {
  color: var(--vp-c-text-1);
}

:global(.dark) .start-header p {
  color: var(--vp-c-text-2);
}

.start-footer {
  text-align: center;
  padding-top: 1.5rem;
}

.start-link {
  display: inline-block;
  padding: 0.875rem 2rem;
  background: var(--vp-c-brand-1);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1.05em;
  transition: all 0.2s ease;
}

.start-link:hover {
  background: var(--vp-c-brand-2);
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
}

/* 暗夜主题链接修复 */
:global(.dark) .start-link {
  background: var(--vp-c-brand-1);
  color: #09090b;
}

:global(.dark) .start-link:hover {
  background: var(--vp-c-brand-2);
  color: #09090b;
}

.outcome-icon {
  font-size: 2em;
  margin-bottom: 0.75rem;
}

.outcome-card h4 {
  font-size: 1.1em;
  font-weight: 600;
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.outcome-card p {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  line-height: 1.5;
  margin: 0;
}

.start-footer {
  text-align: center;
}

.start-link {
  display: inline-block;
  padding: 0.875rem 2rem;
  background: var(--vp-c-brand-1);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1.05em;
  transition: all 0.2s ease;
}

.start-link:hover {
  background: var(--vp-c-brand-2);
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
}

@media (max-width: 768px) {
  .quick-start-section {
    padding: 2rem 0;
  }

  .start-header h2 {
    font-size: 1.8em;
  }

  .terminal-content {
    padding: 1rem;
    font-size: 0.85em;
  }

  .outcomes-grid {
    grid-template-columns: 1fr;
  }
}
</style>
