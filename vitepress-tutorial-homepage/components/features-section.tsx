import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Lightbulb, FlaskConical, Boxes, Play } from "lucide-react"

const features = [
  {
    icon: Lightbulb,
    title: "原理优先",
    description: "理解为什么这样设计，而不只是怎么运行",
  },
  {
    icon: FlaskConical,
    title: "对照实验",
    description: "每个设计选择都通过实验回答不这样做会怎样",
  },
  {
    icon: Boxes,
    title: "模块化",
    description: "6 个独立模块，从基础组件到完整架构",
  },
  {
    icon: Play,
    title: "可运行",
    description: "所有实验可在普通笔记本上运行（无需 GPU）",
  },
]

export function FeaturesSection() {
  return (
    <section id="features" className="border-b border-border/40 bg-muted/30 py-20 md:py-28">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">核心特点</h2>
          <p className="mt-4 text-lg text-muted-foreground">
            模块化的 LLM 训练教案，帮助你理解现代大语言模型的训练原理
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => (
            <Card
              key={feature.title}
              className="border-border/50 bg-card/50 backdrop-blur transition-colors hover:border-primary/50"
            >
              <CardHeader>
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-lg">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm leading-relaxed">{feature.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
