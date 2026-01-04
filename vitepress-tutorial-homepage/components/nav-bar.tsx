"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Menu, X, Github, BookOpen, GraduationCap } from "lucide-react"

export function NavBar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <BookOpen className="h-4 w-4" />
          </div>
          <span className="font-semibold text-foreground">MiniMind Notes</span>
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden items-center gap-6 md:flex">
          <Link href="#features" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
            特性
          </Link>
          <Link href="#learning-path" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
            学习路线
          </Link>
          <Link href="#modules" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
            模块导航
          </Link>
          <Link href="#quick-start" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
            快速开始
          </Link>
        </nav>

        <div className="hidden items-center gap-2 md:flex">
          <Button variant="ghost" size="sm" asChild>
            <Link href="https://github.com/joyehuang/minimind-notes" target="_blank">
              <Github className="mr-2 h-4 w-4" />
              GitHub
            </Link>
          </Button>
          <Button size="sm" asChild>
            <Link href="https://github.com/joyehuang/minimind-notes/blob/master/ROADMAP.md" target="_blank">
              <GraduationCap className="mr-2 h-4 w-4" />
              开始学习
            </Link>
          </Button>
        </div>

        {/* Mobile Menu Button */}
        <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </Button>
      </div>

      {/* Mobile Nav */}
      {isOpen && (
        <div className="border-t border-border/40 bg-background md:hidden">
          <nav className="container mx-auto flex flex-col gap-2 px-4 py-4">
            <Link
              href="#features"
              className="rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
              onClick={() => setIsOpen(false)}
            >
              特性
            </Link>
            <Link
              href="#learning-path"
              className="rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
              onClick={() => setIsOpen(false)}
            >
              学习路线
            </Link>
            <Link
              href="#modules"
              className="rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
              onClick={() => setIsOpen(false)}
            >
              模块导航
            </Link>
            <Link
              href="#quick-start"
              className="rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
              onClick={() => setIsOpen(false)}
            >
              快速开始
            </Link>
            <div className="mt-2 flex flex-col gap-2 border-t border-border pt-4">
              <Button variant="outline" size="sm" asChild>
                <Link href="https://github.com/joyehuang/minimind-notes" target="_blank">
                  <Github className="mr-2 h-4 w-4" />
                  GitHub
                </Link>
              </Button>
              <Button size="sm" asChild>
                <Link href="https://github.com/joyehuang/minimind-notes/blob/master/ROADMAP.md" target="_blank">
                  <GraduationCap className="mr-2 h-4 w-4" />
                  开始学习
                </Link>
              </Button>
            </div>
          </nav>
        </div>
      )}
    </header>
  )
}
