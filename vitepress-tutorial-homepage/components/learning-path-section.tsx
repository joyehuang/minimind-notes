import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Zap, BookOpen, GraduationCap, ArrowRight } from "lucide-react"

const paths = [
  {
    icon: Zap,
    title: "快速体验",
    duration: "30 分钟",
    description: "理解核心设计选择",
    badge: "推荐入门",
    badgeVariant: "default" as const,
    href: "https://github.com/joyehuang/minimind-notes/blob/master/ROADMAP.md#-%E5%BF%AB%E9%80%9F%E4%BD%93%E9%AA%8C-30-%E5%88%86%E9%92%9F",
  },
  {
    icon: BookOpen,
    title: "系统学习",
    duration: "6 小时",
    description: "掌握基础组件",
    badge: "全面学习",
    badgeVariant: "secondary" as const,
    href: "https://github.com/joyehuang/minimind-notes/blob/master/ROADMAP.md#-%E7%B3%BB%E7%BB%9F%E5%AD%A6%E4%B9%A0-6-%E5%B0%8F%E6%97%B6",
  },
  {
    icon: GraduationCap,
    title: "深度掌握",
    duration: "30+ 小时",
    description: "从零训练模型",
    badge: "进阶挑战",
    badgeVariant: "outline" as const,
    href: "https://github.com/joyehuang/minimind-notes/blob/master/ROADMAP.md#-%E6%B7%B1%E5%BA%A6%E6%8E%8C%E6%8F%A1-30-%E5%B0%8F%E6%97%B6",
  },
]

export function LearningPathSection() {
  return (
    <section id="learning-path" className="border-b border-border/40 py-20 md:py-28">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">学习路线</h2>
          <p className="mt-4 text-lg text-muted-foreground">根据你的时间和目标，选择合适的路径</p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {paths.map((path) => (
            <Card
              key={path.title}
              className="group relative overflow-hidden border-border/50 transition-all hover:border-primary/50 hover:shadow-lg"
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 transition-colors group-hover:bg-primary/20">
                    <path.icon className="h-6 w-6 text-primary" />
                  </div>
                  <Badge variant={path.badgeVariant}>{path.badge}</Badge>
                </div>
                <CardTitle className="mt-4 text-xl">{path.title}</CardTitle>
                <CardDescription className="text-sm">{path.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4 flex items-baseline gap-1">
                  <span className="text-3xl font-bold text-foreground">{path.duration}</span>
                </div>
                <Button
                  variant="outline"
                  className="w-full group-hover:bg-primary group-hover:text-primary-foreground bg-transparent"
                  asChild
                >
                  <Link href={path.href} target="_blank">
                    开始学习
                    <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
