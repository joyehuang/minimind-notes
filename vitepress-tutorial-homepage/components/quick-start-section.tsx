import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Terminal, ExternalLink } from "lucide-react"

export function QuickStartSection() {
  return (
    <section id="quick-start" className="border-b border-border/40 py-20 md:py-28">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">快速开始</h2>
          <p className="mt-4 text-lg text-muted-foreground">30 分钟体验核心设计，运行三个关键实验</p>
        </div>

        <div className="mx-auto max-w-3xl">
          <div className="rounded-xl border border-border/50 bg-card overflow-hidden">
            {/* Terminal Header */}
            <div className="flex items-center gap-2 border-b border-border/50 bg-muted/50 px-4 py-3">
              <div className="flex gap-1.5">
                <div className="h-3 w-3 rounded-full bg-red-500/80" />
                <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
                <div className="h-3 w-3 rounded-full bg-green-500/80" />
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Terminal className="h-4 w-4" />
                <span>Terminal</span>
              </div>
            </div>

            {/* Code Content */}
            <div className="p-5 font-mono text-sm">
              <div className="space-y-4">
                <div>
                  <p className="text-muted-foreground"># 1. 克隆仓库</p>
                  <p className="text-foreground">git clone https://github.com/joyehuang/minimind-notes.git</p>
                  <p className="text-foreground">cd minimind-notes</p>
                </div>

                <div>
                  <p className="text-muted-foreground"># 2. 激活虚拟环境（如果已有）</p>
                  <p className="text-foreground">source venv/bin/activate</p>
                </div>

                <div>
                  <p className="text-muted-foreground"># 3. 实验1：为什么需要归一化？</p>
                  <p className="text-foreground">cd modules/01-foundation/01-normalization/experiments</p>
                  <p className="text-emerald-500">python exp1_gradient_vanishing.py</p>
                </div>

                <div>
                  <p className="text-muted-foreground"># 4. 实验2：为什么用 RoPE 位置编码？</p>
                  <p className="text-foreground">cd ../../02-position-encoding/experiments</p>
                  <p className="text-emerald-500">python exp1_rope_basics.py</p>
                </div>

                <div>
                  <p className="text-muted-foreground"># 5. 实验3：Attention 如何工作？</p>
                  <p className="text-foreground">cd ../../03-attention/experiments</p>
                  <p className="text-emerald-500">python exp1_attention_basics.py</p>
                </div>
              </div>
            </div>
          </div>

          {/* What you will see */}
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-border/50 bg-card p-4">
              <div className="mb-2 text-2xl">📊</div>
              <h4 className="font-medium text-foreground">梯度消失</h4>
              <p className="mt-1 text-sm text-muted-foreground">可视化深层网络的梯度流动问题</p>
            </div>
            <div className="rounded-lg border border-border/50 bg-card p-4">
              <div className="mb-2 text-2xl">🔄</div>
              <h4 className="font-medium text-foreground">RoPE 编码</h4>
              <p className="mt-1 text-sm text-muted-foreground">旋转位置编码的数学原理演示</p>
            </div>
            <div className="rounded-lg border border-border/50 bg-card p-4">
              <div className="mb-2 text-2xl">🎯</div>
              <h4 className="font-medium text-foreground">Attention</h4>
              <p className="mt-1 text-sm text-muted-foreground">注意力权重的计算过程可视化</p>
            </div>
          </div>

          <div className="mt-8 text-center">
            <Button size="lg" asChild>
              <Link href="https://github.com/joyehuang/minimind-notes/blob/master/ROADMAP.md" target="_blank">
                查看完整学习路线
                <ExternalLink className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
