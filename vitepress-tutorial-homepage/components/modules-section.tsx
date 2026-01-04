import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, Circle, Clock } from "lucide-react"

const tier1Modules = [
  {
    id: "01-normalization",
    title: "归一化",
    question: "为什么要归一化？Pre-LN vs Post-LN？",
    experiments: 2,
    status: "complete",
    href: "https://github.com/joyehuang/minimind-notes/tree/master/modules/01-foundation/01-normalization",
  },
  {
    id: "02-position-encoding",
    title: "位置编码",
    question: "为什么选择 RoPE？如何长度外推？",
    experiments: 4,
    status: "partial",
    href: "https://github.com/joyehuang/minimind-notes/tree/master/modules/01-foundation/02-position-encoding",
  },
  {
    id: "03-attention",
    title: "注意力机制",
    question: "QKV 的直觉是什么？为什么多头？",
    experiments: 3,
    status: "partial",
    href: "https://github.com/joyehuang/minimind-notes/tree/master/modules/01-foundation/03-attention",
  },
  {
    id: "04-feedforward",
    title: "前馈网络",
    question: "FFN 存储什么知识？为什么扩张？",
    experiments: 1,
    status: "partial",
    href: "https://github.com/joyehuang/minimind-notes/tree/master/modules/01-foundation/04-feedforward",
  },
]

const tier2Modules = [
  {
    id: "01-residual-connection",
    title: "残差连接",
    question: "为什么需要残差？如何稳定梯度？",
    status: "planned",
    href: "https://github.com/joyehuang/minimind-notes/tree/master/modules/02-architecture/01-residual-connection",
  },
  {
    id: "02-transformer-block",
    title: "Transformer Block",
    question: "如何组装组件？为什么这个顺序？",
    status: "planned",
    href: "https://github.com/joyehuang/minimind-notes/tree/master/modules/02-architecture/02-transformer-block",
  },
]

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case "complete":
      return (
        <Badge variant="default" className="gap-1 bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500/20">
          <CheckCircle2 className="h-3 w-3" />
          完整
        </Badge>
      )
    case "partial":
      return (
        <Badge variant="secondary" className="gap-1 bg-amber-500/10 text-amber-600 hover:bg-amber-500/20">
          <Circle className="h-3 w-3" />
          实验完成
        </Badge>
      )
    case "planned":
      return (
        <Badge variant="outline" className="gap-1">
          <Clock className="h-3 w-3" />
          待开发
        </Badge>
      )
    default:
      return null
  }
}

export function ModulesSection() {
  return (
    <section id="modules" className="border-b border-border/40 bg-muted/30 py-20 md:py-28">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">模块导航</h2>
          <p className="mt-4 text-lg text-muted-foreground">从基础组件到完整架构的模块化学习</p>
        </div>

        {/* Tier 1 */}
        <div className="mb-12">
          <div className="mb-6 flex items-center gap-3">
            <Badge variant="default" className="px-3 py-1 text-sm">
              Tier 1
            </Badge>
            <h3 className="text-xl font-semibold text-foreground">Foundation（基础组件）</h3>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {tier1Modules.map((module) => (
              <Link
                key={module.id}
                href={module.href}
                target="_blank"
                className="group flex flex-col rounded-lg border border-border/50 bg-card p-5 transition-all hover:border-primary/50 hover:shadow-md"
              >
                <div className="mb-3 flex items-center justify-between">
                  <span className="font-mono text-sm text-muted-foreground">{module.id}</span>
                  <StatusBadge status={module.status} />
                </div>
                <h4 className="mb-2 text-lg font-semibold text-foreground group-hover:text-primary">{module.title}</h4>
                <p className="mb-3 flex-1 text-sm text-muted-foreground">{module.question}</p>
                {module.experiments && (
                  <div className="text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">{module.experiments}</span> 个实验
                  </div>
                )}
              </Link>
            ))}
          </div>
        </div>

        {/* Tier 2 */}
        <div>
          <div className="mb-6 flex items-center gap-3">
            <Badge variant="secondary" className="px-3 py-1 text-sm">
              Tier 2
            </Badge>
            <h3 className="text-xl font-semibold text-foreground">Architecture（架构组装）</h3>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {tier2Modules.map((module) => (
              <Link
                key={module.id}
                href={module.href}
                target="_blank"
                className="group flex flex-col rounded-lg border border-border/50 bg-card p-5 transition-all hover:border-primary/50 hover:shadow-md"
              >
                <div className="mb-3 flex items-center justify-between">
                  <span className="font-mono text-sm text-muted-foreground">{module.id}</span>
                  <StatusBadge status={module.status} />
                </div>
                <h4 className="mb-2 text-lg font-semibold text-foreground group-hover:text-primary">{module.title}</h4>
                <p className="flex-1 text-sm text-muted-foreground">{module.question}</p>
              </Link>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="mt-8 flex flex-wrap justify-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <span>完整：包含教学文档 + 实验代码 + 自测题</span>
          </div>
          <div className="flex items-center gap-2">
            <Circle className="h-4 w-4 text-amber-500" />
            <span>实验完成：有实验代码，文档待补充</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span>待开发：仅目录结构</span>
          </div>
        </div>
      </div>
    </section>
  )
}
