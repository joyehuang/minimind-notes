import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { ModulesSection } from "@/components/modules-section"
import { LearningPathSection } from "@/components/learning-path-section"
import { QuickStartSection } from "@/components/quick-start-section"
import { FooterSection } from "@/components/footer-section"
import { NavBar } from "@/components/nav-bar"

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main>
        <HeroSection />
        <FeaturesSection />
        <LearningPathSection />
        <ModulesSection />
        <QuickStartSection />
      </main>
      <FooterSection />
    </div>
  )
}
