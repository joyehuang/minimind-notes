<template>
  <section class="home-hero-video" :aria-label="ariaLabel">
    <div class="video-frame">
      <video
        :key="videoSrc"
        class="hero-video"
        autoplay
        muted
        loop
        playsinline
        preload="metadata"
      >
        <source :src="videoSrc" type="video/mp4" />
        {{ fallbackText }}
      </video>
      <div class="video-glow" aria-hidden="true"></div>
    </div>
    <p class="video-caption">{{ caption }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vitepress";

const route = useRoute();

const isEnglishRoute = computed(() => route.path === "/en/" || route.path.startsWith("/en/"));

const videoSrc = computed(() =>
  isEnglishRoute.value ? "/videos/home-hero.mp4" : "/videos/home-hero-zh.mp4"
);

const caption = computed(() =>
  isEnglishRoute.value
    ? "A 30 second overview: modular lessons, experiments, and roadmap."
    : "30 秒概览：模块化课程、实验与学习路线图。"
);

const fallbackText = computed(() =>
  isEnglishRoute.value
    ? "Your browser does not support the video tag."
    : "你的浏览器不支持视频播放。"
);

const ariaLabel = computed(() =>
  isEnglishRoute.value ? "minimind homepage intro video" : "minimind 首页介绍视频"
);
</script>

<style scoped>
.home-hero-video {
  margin: 2.5rem auto 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 0 1.5rem;
}

.video-frame {
  position: relative;
  width: min(100%, 1040px);
  aspect-ratio: 16 / 9;
  border-radius: 20px;
  overflow: hidden;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  box-shadow: 0 22px 40px rgba(15, 23, 42, 0.18);
}

.hero-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.video-glow {
  position: absolute;
  inset: -30% -20% 30% -20%;
  background: radial-gradient(
    circle,
    rgba(56, 189, 248, 0.25) 0%,
    rgba(15, 23, 42, 0) 70%
  );
  opacity: 0.4;
  pointer-events: none;
}

.video-caption {
  max-width: 720px;
  text-align: center;
  color: var(--vp-c-text-2);
  font-size: 0.95rem;
  line-height: 1.6;
}

@media (max-width: 768px) {
  .home-hero-video {
    margin-top: 2rem;
  }

  .video-frame {
    border-radius: 16px;
  }
}
</style>
