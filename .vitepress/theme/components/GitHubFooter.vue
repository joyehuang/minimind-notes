<template>
  <div class="github-footer">
    <p class="github-footer-text">
      发现错误或有疑问？欢迎在 
      <a 
        href="https://github.com/joyehuang/minimind-notes" 
        target="_blank" 
        rel="noopener noreferrer"
        class="github-link"
      >
        GitHub
      </a>
      提 Issue 或 Star 支持我们！
    </p>
  </div>
</template>

<script setup lang="ts">
// GitHub 页脚组件
</script>

<style scoped>
.github-footer {
  margin-top: 3rem;
  margin-bottom: 1rem;
  padding: 1.25rem 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  text-align: center;
  transition: all 0.2s ease;
}

.github-footer:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.github-footer-text {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  line-height: 1.6;
}

.github-link {
  color: var(--vp-c-brand-1);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s;
  position: relative;
}

.github-link:hover {
  color: var(--vp-c-brand-2);
  text-decoration: underline;
}

/* 暗色模式优化 */
:global(.dark) .github-footer {
  background: var(--vp-c-accent-bg);
  border-color: var(--vp-c-divider);
}

:global(.dark) .github-footer:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

:global(.dark) .github-footer-text {
  color: var(--vp-c-text-1);
}

/* 移动端优化 */
@media (max-width: 768px) {
  .github-footer {
    padding: 1rem;
    margin-top: 2rem;
  }
  
  .github-footer-text {
    font-size: 0.85rem;
  }
}
</style>