/**
 * InteractiveQuiz component translations
 * Extracted to module-level constants to reduce memory overhead
 * and avoid unnecessary reactivity tracking
 */

export interface QuizTranslations {
  multiple: string
  explanation: string
  submit: string
  completeAll: string
  retry: string
  resultsTitle: string
  accuracy: string
  feedback: {
    perfectTitle: string
    perfectText: string
    greatTitle: string
    greatText: string
    goodTitle: string
    goodText: string
    needsWorkTitle: string
    needsWorkText: string
  }
}

export const QUIZ_TRANSLATIONS = {
  en: {
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
      greatText: 'You\'ve mastered most of the concepts. Review the mistakes and keep going.',
      goodTitle: 'Keep it up!',
      goodText: 'You understand the basics but need more depth. Revisit teaching.md and retry.',
      needsWorkTitle: 'Needs Work',
      needsWorkText: 'Go back to teaching.md, run the experiments, then try again — take it slow!'
    }
  } as QuizTranslations,
  zh: {
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
  } as QuizTranslations
} as const
