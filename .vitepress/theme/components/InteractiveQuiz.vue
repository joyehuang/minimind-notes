<template>
  <div class="interactive-quiz">
    <div v-for="(question, qIndex) in questions" :key="qIndex" class="quiz-question">
      <div class="question-header">
        <span class="question-number">Q{{ qIndex + 1 }}</span>
        <h3 class="question-title">{{ question.question }}</h3>
        <span v-if="question.type === 'multiple'" class="question-badge">{{ copy.multiple }}</span>
      </div>

      <div class="options">
        <div
          v-for="(option, oIndex) in question.options"
          :key="oIndex"
          class="option"
          :class="{
            'option-selected': isSelected(qIndex, oIndex),
            'option-correct': submitted && isCorrectOption(qIndex, oIndex),
            'option-incorrect': submitted && isSelected(qIndex, oIndex) && !isCorrectOption(qIndex, oIndex),
            'option-disabled': submitted
          }"
          @click="!submitted && selectOption(qIndex, oIndex, question.type)"
        >
          <div class="option-marker">
            <template v-if="question.type === 'multiple'">
              <input
                type="checkbox"
                :checked="isSelected(qIndex, oIndex)"
                :disabled="submitted"
                @click.stop
              />
            </template>
            <template v-else>
              <input
                type="radio"
                :name="`question-${qIndex}`"
                :checked="isSelected(qIndex, oIndex)"
                :disabled="submitted"
                @click.stop
              />
            </template>
          </div>
          <div class="option-label">{{ option.label }}</div>
          <div class="option-text">{{ option.text }}</div>
          <div v-if="submitted && isCorrectOption(qIndex, oIndex)" class="option-icon">✓</div>
          <div v-if="submitted && isSelected(qIndex, oIndex) && !isCorrectOption(qIndex, oIndex)" class="option-icon">✗</div>
        </div>
      </div>

      <div v-if="submitted" class="explanation">
        <div class="explanation-header">
          <span class="explanation-icon">💡</span>
          <strong>{{ copy.explanation }}</strong>
        </div>
        <div class="explanation-content" v-html="question.explanation"></div>
      </div>
    </div>

    <div class="quiz-actions">
      <button
        v-if="!submitted"
        class="submit-button"
        :disabled="!canSubmit"
        @click="submitQuiz"
      >
        {{ canSubmit ? copy.submit : copy.completeAll }}
      </button>
      <button v-else class="reset-button" @click="resetQuiz">
        {{ copy.retry }}
      </button>
    </div>

    <div v-if="submitted" class="quiz-results">
      <div class="results-header">
        <h3>📊 {{ copy.resultsTitle }}</h3>
      </div>
      <div class="results-score">
        <div class="score-circle" :class="scoreLevel">
          <div class="score-value">{{ score }}</div>
          <div class="score-total">/ {{ questions.length }}</div>
        </div>
        <div class="score-percentage">{{ copy.accuracy }}: {{ scorePercentage }}%</div>
      </div>
      <div class="results-feedback">
        <div v-if="scorePercentage === 100" class="feedback feedback-perfect">
          <div class="feedback-icon">🎉</div>
          <div class="feedback-title">{{ copy.feedback.perfectTitle }}</div>
          <div class="feedback-text">
            {{ copy.feedback.perfectText }}
          </div>
        </div>
        <div v-else-if="scorePercentage >= 80" class="feedback feedback-great">
          <div class="feedback-icon">👍</div>
          <div class="feedback-title">{{ copy.feedback.greatTitle }}</div>
          <div class="feedback-text">
            {{ copy.feedback.greatText }}
          </div>
        </div>
        <div v-else-if="scorePercentage >= 60" class="feedback feedback-good">
          <div class="feedback-icon">💪</div>
          <div class="feedback-title">{{ copy.feedback.goodTitle }}</div>
          <div class="feedback-text">
            {{ copy.feedback.goodText }}
          </div>
        </div>
        <div v-else class="feedback feedback-needs-work">
          <div class="feedback-icon">📚</div>
          <div class="feedback-title">{{ copy.feedback.needsWorkTitle }}</div>
          <div class="feedback-text">
            {{ copy.feedback.needsWorkText }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useLocale } from '../i18n'

const { isEn } = useLocale()

const copy = computed(() =>
  isEn.value
    ? {
        multiple: 'Multiple',
        explanation: 'Explanation',
        submit: 'Submit Answers',
        completeAll: 'Please answer all questions',
        retry: 'Retry',
        resultsTitle: 'Quiz Results',
        accuracy: 'Accuracy',
        feedback: {
          perfectTitle: 'Perfect!',
          perfectText: 'You got everything right. Solid understanding — move on to the next module!',
          greatTitle: 'Great!',
          greatText: 'You’ve mastered most of the concepts. Review the mistakes and keep going.',
          goodTitle: 'Keep it up!',
          goodText: 'You understand the basics but need more depth. Revisit teaching.md and retry.',
          needsWorkTitle: 'Needs Work',
          needsWorkText: 'Go back to teaching.md, run the experiments, then try again — take it slow!'
        }
      }
    : {
        multiple: '多选',
        explanation: '解析',
        submit: '提交答案',
        completeAll: '请完成所有题目',
        retry: '重新答题',
        resultsTitle: '答题统计',
        accuracy: '正确率',
        feedback: {
          perfectTitle: '完美!',
          perfectText: '恭喜你全部答对！你对本模块的理解非常扎实。可以继续学习下一个模块了！',
          greatTitle: '很好!',
          greatText: '你已经掌握了大部分知识点。建议复习一下错题，巩固理解后继续前进。',
          goodTitle: '继续加油!',
          goodText: '你已经理解了基础概念，但还需要加深理解。建议回到 teaching.md 复习一下，然后重新测试。',
          needsWorkTitle: '需要加强',
          needsWorkText: '建议先回到 teaching.md 系统学习一遍，运行实验代码加深理解，然后再来测试。不要着急，慢慢来！'
        }
      }
)

interface QuizOption {
  label: string
  text: string
}

interface QuizQuestion {
  question: string
  type: 'single' | 'multiple'
  options: QuizOption[]
  correct: number[] // Indices of correct options
  explanation: string
}

interface Props {
  questions: QuizQuestion[]
  quizId?: string
}

const props = withDefaults(defineProps<Props>(), {
  quizId: 'default-quiz'
})

const userAnswers = ref<number[][]>([])
const submitted = ref(false)

const storageKey = computed(() => `quiz-${props.quizId}-answers`)

onMounted(() => {
  // Initialize empty answers
  userAnswers.value = props.questions.map(() => [])

  // Defer localStorage access to avoid blocking initial render
  const loadSavedState = () => {
    try {
      const saved = localStorage.getItem(storageKey.value)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed.submitted) {
          userAnswers.value = parsed.answers
          submitted.value = true
        }
      }
    } catch (e) {
      // Ignore localStorage errors
    }
  }

  // Load state during idle time
  if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
    requestIdleCallback(loadSavedState)
  } else {
    setTimeout(loadSavedState, 0)
  }
})

const isSelected = (qIndex: number, oIndex: number): boolean => {
  return userAnswers.value[qIndex]?.includes(oIndex) || false
}

const isCorrectOption = (qIndex: number, oIndex: number): boolean => {
  return props.questions[qIndex].correct.includes(oIndex)
}

const selectOption = (qIndex: number, oIndex: number, type: string) => {
  if (type === 'multiple') {
    // Multiple choice: toggle selection
    if (isSelected(qIndex, oIndex)) {
      userAnswers.value[qIndex] = userAnswers.value[qIndex].filter(i => i !== oIndex)
    } else {
      userAnswers.value[qIndex] = [...(userAnswers.value[qIndex] || []), oIndex]
    }
  } else {
    // Single choice: replace selection
    userAnswers.value[qIndex] = [oIndex]
  }
}

const canSubmit = computed(() => {
  return userAnswers.value.every(answer => answer.length > 0)
})

const score = computed(() => {
  let correct = 0
  props.questions.forEach((q, i) => {
    const userAnswer = userAnswers.value[i] || []
    const correctAnswer = q.correct

    // Check if answers match (order doesn't matter)
    if (
      userAnswer.length === correctAnswer.length &&
      userAnswer.every(a => correctAnswer.includes(a))
    ) {
      correct++
    }
  })
  return correct
})

const scorePercentage = computed(() => {
  return Math.round((score.value / props.questions.length) * 100)
})

const scoreLevel = computed(() => {
  const pct = scorePercentage.value
  if (pct === 100) return 'perfect'
  if (pct >= 80) return 'great'
  if (pct >= 60) return 'good'
  return 'needs-work'
})

const submitQuiz = () => {
  submitted.value = true

  // Save to localStorage
  try {
    localStorage.setItem(storageKey.value, JSON.stringify({
      answers: userAnswers.value,
      submitted: true,
      timestamp: Date.now()
    }))
  } catch (e) {
    // Ignore localStorage errors
  }
}

const resetQuiz = () => {
  userAnswers.value = props.questions.map(() => [])
  submitted.value = false

  // Clear localStorage
  try {
    localStorage.removeItem(storageKey.value)
  } catch (e) {
    // Ignore localStorage errors
  }
}
</script>

<style scoped>
.interactive-quiz {
  margin: 2rem 0;
}

.quiz-question {
  margin-bottom: 2.5rem;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

.question-header {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.question-number {
  flex-shrink: 0;
  padding: 0.25rem 0.6rem;
  background: var(--vp-c-brand-1);
  color: white;
  border-radius: 4px;
  font-weight: bold;
  font-size: 0.9em;
}

.question-title {
  flex: 1;
  margin: 0;
  font-size: 1.1em;
  color: var(--vp-c-text-1);
}

.question-badge {
  flex-shrink: 0;
  padding: 0.25rem 0.6rem;
  background: var(--vp-c-warning-soft);
  color: var(--vp-c-warning-1);
  border-radius: 4px;
  font-size: 0.8em;
  font-weight: 500;
}

.options {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.option {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--vp-c-bg);
  border: 2px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.option:hover:not(.option-disabled) {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 2px 8px rgba(var(--vp-c-brand-rgb), 0.1);
}

.option-selected:not(.option-disabled) {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.option-correct {
  border-color: #10b981 !important;
  background: rgba(16, 185, 129, 0.1) !important;
}

.option-incorrect {
  border-color: #ef4444 !important;
  background: rgba(239, 68, 68, 0.1) !important;
}

.option-disabled {
  cursor: not-allowed;
  opacity: 0.8;
}

.option-marker {
  flex-shrink: 0;
}

.option-marker input {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.option-disabled input {
  cursor: not-allowed;
}

.option-label {
  flex-shrink: 0;
  font-weight: bold;
  color: var(--vp-c-text-1);
  min-width: 1.5rem;
}

.option-text {
  flex: 1;
  color: var(--vp-c-text-2);
}

.option-icon {
  flex-shrink: 0;
  font-size: 1.2em;
  font-weight: bold;
}

.option-correct .option-icon {
  color: #10b981;
}

.option-incorrect .option-icon {
  color: #ef4444;
}

.explanation {
  margin-top: 1.5rem;
  padding: 1rem;
  background: var(--vp-c-bg);
  border-left: 4px solid var(--vp-c-brand-1);
  border-radius: 4px;
}

.explanation-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.explanation-icon {
  font-size: 1.2em;
}

.explanation-content {
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.explanation-content :deep(ul) {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.explanation-content :deep(code) {
  padding: 0.2rem 0.4rem;
  background: var(--vp-c-bg-soft);
  border-radius: 3px;
  font-size: 0.9em;
}

.quiz-actions {
  display: flex;
  justify-content: center;
  margin: 2rem 0;
}

.submit-button,
.reset-button {
  padding: 0.75rem 2rem;
  font-size: 1.05em;
  font-weight: 600;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.submit-button {
  background: var(--vp-c-brand-1);
  color: white;
}

.submit-button:hover:not(:disabled) {
  background: var(--vp-c-brand-2);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(var(--vp-c-brand-rgb), 0.3);
}

.submit-button:disabled {
  background: var(--vp-c-divider);
  color: var(--vp-c-text-3);
  cursor: not-allowed;
}

.reset-button {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  border: 2px solid var(--vp-c-divider);
}

.reset-button:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.quiz-results {
  margin-top: 2rem;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 2px solid var(--vp-c-divider);
}

.results-header h3 {
  margin: 0 0 1.5rem 0;
  text-align: center;
  color: var(--vp-c-brand-1);
  font-size: 1.5em;
}

.results-score {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 2rem;
}

.score-circle {
  position: relative;
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 6px solid;
  margin-bottom: 1rem;
}

.score-circle.perfect {
  border-color: #10b981;
  background: rgba(16, 185, 129, 0.1);
}

.score-circle.great {
  border-color: #3b82f6;
  background: rgba(59, 130, 246, 0.1);
}

.score-circle.good {
  border-color: #f59e0b;
  background: rgba(245, 158, 11, 0.1);
}

.score-circle.needs-work {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
}

.score-value {
  font-size: 2.5em;
  font-weight: bold;
  line-height: 1;
}

.score-total {
  font-size: 1.2em;
  color: var(--vp-c-text-2);
}

.score-percentage {
  font-size: 1.2em;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.results-feedback {
  margin-top: 1.5rem;
}

.feedback {
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 4px solid;
}

.feedback-perfect {
  background: rgba(16, 185, 129, 0.1);
  border-color: #10b981;
}

.feedback-great {
  background: rgba(59, 130, 246, 0.1);
  border-color: #3b82f6;
}

.feedback-good {
  background: rgba(245, 158, 11, 0.1);
  border-color: #f59e0b;
}

.feedback-needs-work {
  background: rgba(239, 68, 68, 0.1);
  border-color: #ef4444;
}

.feedback-icon {
  font-size: 2em;
  text-align: center;
  margin-bottom: 0.5rem;
}

.feedback-title {
  font-size: 1.3em;
  font-weight: bold;
  text-align: center;
  margin-bottom: 0.75rem;
  color: var(--vp-c-text-1);
}

.feedback-text {
  color: var(--vp-c-text-2);
  line-height: 1.6;
  text-align: center;
}

@media (max-width: 768px) {
  .quiz-question {
    padding: 1rem;
  }

  .question-header {
    flex-wrap: wrap;
  }

  .option {
    padding: 0.75rem;
  }

  .quiz-results {
    padding: 1.5rem 1rem;
  }
}
</style>
