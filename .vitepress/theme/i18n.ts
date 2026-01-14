import { computed } from 'vue'
import { useData } from 'vitepress'

export const useLocale = () => {
  const { lang } = useData()
  const isEn = computed(() => (lang.value || '').toLowerCase().startsWith('en'))
  const localePath = computed(() => (isEn.value ? '/en' : ''))

  return {
    lang,
    isEn,
    localePath
  }
}
