import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'

const root = process.cwd()
const requiredPages = ['index.md', 'teaching.md', 'code_guide.md', 'quiz.md']
const completeModules = [
  'modules/01-foundation/01-normalization',
  'modules/01-foundation/02-position-encoding',
  'modules/01-foundation/03-attention',
  'modules/01-foundation/04-feedforward',
  'modules/02-architecture/01-residual-connection',
  'modules/02-architecture/02-transformer-block',
  'en/modules/01-foundation/01-normalization',
  'en/modules/01-foundation/02-position-encoding',
  'en/modules/01-foundation/03-attention',
  'en/modules/01-foundation/04-feedforward',
  'en/modules/02-architecture/01-residual-connection',
  'en/modules/02-architecture/02-transformer-block'
]

const failures = []

async function exists(filePath) {
  try {
    await stat(filePath)
    return true
  } catch {
    return false
  }
}

function routeCandidates(route) {
  const relative = route.replace(/^\/+/, '').replace(/\/+$/, '')
  return [
    path.join(root, `${relative}.md`),
    path.join(root, relative, 'index.md'),
    path.join(root, relative)
  ]
}

function relativeCandidates(sourceFile, target) {
  const resolved = path.resolve(path.dirname(sourceFile), target)
  return [
    resolved,
    `${resolved}.md`,
    path.join(resolved, 'index.md')
  ]
}

async function validateLink(sourceFile, rawTarget) {
  const target = rawTarget.split('#')[0].split('?')[0]
  if (
    !target ||
    target.startsWith('#') ||
    target.startsWith('http://') ||
    target.startsWith('https://') ||
    target.startsWith('mailto:')
  ) {
    return
  }

  const candidates = target.startsWith('/')
    ? routeCandidates(target)
    : relativeCandidates(sourceFile, target)

  if (!(await Promise.all(candidates.map(exists))).some(Boolean)) {
    failures.push(`${path.relative(root, sourceFile)} -> ${rawTarget}`)
  }
}

async function validateMarkdownLinks(filePath, content) {
  const linkPattern = /(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g
  for (const match of content.matchAll(linkPattern)) {
    await validateLink(filePath, match[1])
  }
}

for (const moduleDir of completeModules) {
  const absoluteDir = path.join(root, moduleDir)

  for (const page of requiredPages) {
    const filePath = path.join(absoluteDir, page)
    if (!(await exists(filePath))) {
      failures.push(`${moduleDir} 缺少 ${page}`)
      continue
    }

    const content = await readFile(filePath, 'utf8')
    if (/This module is planned|本模块待开发|仅目录结构/.test(content)) {
      failures.push(`${path.relative(root, filePath)} 仍包含占位内容`)
    }
    await validateMarkdownLinks(filePath, content)
  }

  if (!moduleDir.startsWith('en/')) {
    const experimentsDir = path.join(absoluteDir, 'experiments')
    if (!(await exists(experimentsDir))) {
      failures.push(`${moduleDir} 缺少 experiments/`)
    } else {
      const experimentFiles = (await readdir(experimentsDir)).filter((name) =>
        /^exp\d+_.+\.py$/.test(name)
      )
      if (experimentFiles.length === 0) {
        failures.push(`${moduleDir} 没有可运行实验`)
      }
    }
  }
}

if (failures.length > 0) {
  console.error('模块内容校验失败：')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log(`模块内容校验通过：${completeModules.length} 个中英文模块`)
