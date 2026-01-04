import Link from "next/link"
import { Github, Heart, ExternalLink } from "lucide-react"

export function FooterSection() {
  return (
    <footer className="border-t border-border/40 bg-muted/30">
      <div className="container mx-auto max-w-6xl px-4 py-12">
        <div className="grid gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="md:col-span-2">
            <h3 className="text-lg font-semibold text-foreground">MiniMind Notes</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              模块化的 LLM 训练教案，帮助你理解现代大语言模型的训练原理。
            </p>
            <p className="mt-4 flex items-center gap-1 text-sm text-muted-foreground">
              基于{" "}
              <Link
                href="https://github.com/jingyaogong/minimind"
                target="_blank"
                className="inline-flex items-center gap-1 font-medium text-foreground hover:text-primary"
              >
                MiniMind
                <ExternalLink className="h-3 w-3" />
              </Link>
            </p>
          </div>

          {/* Resources */}
          <div>
            <h4 className="mb-4 font-semibold text-foreground">相关资源</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link
                  href="https://arxiv.org/abs/1706.03762"
                  target="_blank"
                  className="text-muted-foreground hover:text-foreground"
                >
                  Attention Is All You Need
                </Link>
              </li>
              <li>
                <Link
                  href="https://arxiv.org/abs/2104.09864"
                  target="_blank"
                  className="text-muted-foreground hover:text-foreground"
                >
                  RoFormer: RoPE
                </Link>
              </li>
              <li>
                <Link
                  href="https://jalammar.github.io/illustrated-transformer/"
                  target="_blank"
                  className="text-muted-foreground hover:text-foreground"
                >
                  The Illustrated Transformer
                </Link>
              </li>
              <li>
                <Link
                  href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
                  target="_blank"
                  className="text-muted-foreground hover:text-foreground"
                >
                  Let's build GPT - Karpathy
                </Link>
              </li>
            </ul>
          </div>

          {/* Links */}
          <div>
            <h4 className="mb-4 font-semibold text-foreground">链接</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link
                  href="https://github.com/joyehuang/minimind-notes"
                  target="_blank"
                  className="flex items-center gap-2 text-muted-foreground hover:text-foreground"
                >
                  <Github className="h-4 w-4" />
                  GitHub
                </Link>
              </li>
              <li>
                <Link
                  href="https://github.com/joyehuang/minimind-notes/issues"
                  target="_blank"
                  className="text-muted-foreground hover:text-foreground"
                >
                  问题反馈
                </Link>
              </li>
              <li>
                <Link
                  href="https://github.com/joyehuang/minimind-notes/blob/master/LICENSE"
                  target="_blank"
                  className="text-muted-foreground hover:text-foreground"
                >
                  Apache-2.0 License
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border/40 pt-8 sm:flex-row">
          <p className="text-sm text-muted-foreground">© 2026 MiniMind Notes. 开源于 GitHub。</p>
          <p className="flex items-center gap-1 text-sm text-muted-foreground">
            Made with <Heart className="h-4 w-4 text-red-500" /> for the AI community
          </p>
        </div>
      </div>
    </footer>
  )
}
