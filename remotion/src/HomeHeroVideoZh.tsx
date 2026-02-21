import React from "react";
import {
  AbsoluteFill,
  Series,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/NotoSansSC";

const { fontFamily } = loadFont("normal", {
  ignoreTooManyRequestsWarning: true,
});

const CODE_FONT =
  "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace";

const THEME = {
  bg: "#0b1020",
  bgAlt: "#0f172a",
  text: "#f8fafc",
  muted: "rgba(248, 250, 252, 0.7)",
  accentBlue: "#38bdf8",
  accentViolet: "#a78bfa",
  accentAmber: "#f59e0b",
  accentEmerald: "#34d399",
  panel: "rgba(10, 16, 32, 0.78)",
  panelBorder: "rgba(148, 163, 184, 0.22)",
  codeKeyword: "#38bdf8",
  codeString: "#facc15",
  codeValue: "#34d399",
  codeComment: "rgba(148, 163, 184, 0.8)",
};

const SPRING_SMOOTH = { damping: 200 };

export type HomeHeroVideoZhProps = {
  brandName: string;
  website: string;
};

export const homeHeroVideoZhProps: HomeHeroVideoZhProps = {
  brandName: "minimind",
  website: "minimind.wiki",
};

const sceneSeconds = 3;

export const HomeHeroVideoZh: React.FC<HomeHeroVideoZhProps> = ({
  brandName,
  website,
}) => {
  const { fps } = useVideoConfig();
  const sceneDuration = sceneSeconds * fps;
  const premount = Math.round(0.4 * fps);

  return (
    <AbsoluteFill style={{ backgroundColor: THEME.bg, fontFamily }}>
      <Series>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <IntroScene
            durationInFrames={sceneDuration}
            brandName={brandName}
            website={website}
          />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <FocusScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <LearningPathScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <ExperimentsScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <BuildScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <AssetsScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <WorkflowScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <NotesScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <OutcomesScene durationInFrames={sceneDuration} />
        </Series.Sequence>
        <Series.Sequence durationInFrames={sceneDuration} premountFor={premount}>
          <CtaScene
            durationInFrames={sceneDuration}
            brandName={brandName}
            website={website}
          />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};

const SceneShell: React.FC<{
  durationInFrames: number;
  accent: string;
  children: React.ReactNode;
}> = ({ durationInFrames, accent, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const fadeIn = interpolate(frame, [0, 0.35 * fps], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 0.4 * fps, durationInFrames],
    [1, 0],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );
  const opacity = Math.min(fadeIn, fadeOut);

  return (
    <AbsoluteFill style={{ color: THEME.text, opacity }}>
      <Background accent={accent} durationInFrames={durationInFrames} />
      <CodingOverlay durationInFrames={durationInFrames} accent={accent} />
      <div
        style={{
          position: "relative",
          zIndex: 2,
          height: "100%",
          padding: "56px 80px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 20,
        }}
      >
        {children}
      </div>
    </AbsoluteFill>
  );
};

const Background: React.FC<{
  accent: string;
  durationInFrames: number;
}> = ({ accent, durationInFrames }) => {
  const frame = useCurrentFrame();

  const orb1X = interpolate(frame, [0, durationInFrames], [-120, 120], {
    extrapolateRight: "clamp",
  });
  const orb1Y = interpolate(frame, [0, durationInFrames], [-80, 60], {
    extrapolateRight: "clamp",
  });
  const orb2X = interpolate(frame, [0, durationInFrames], [160, -100], {
    extrapolateRight: "clamp",
  });
  const orb2Y = interpolate(frame, [0, durationInFrames], [110, -70], {
    extrapolateRight: "clamp",
  });
  const orb3X = interpolate(frame, [0, durationInFrames], [-60, 90], {
    extrapolateRight: "clamp",
  });
  const orb3Y = interpolate(frame, [0, durationInFrames], [140, -90], {
    extrapolateRight: "clamp",
  });
  const gridShift = interpolate(frame, [0, durationInFrames], [0, 12], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: THEME.bg, overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          backgroundImage: `radial-gradient(circle at 20% 20%, ${accent}33 0%, transparent 55%), radial-gradient(circle at 80% 80%, ${THEME.accentViolet}22 0%, transparent 60%)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 520,
          height: 520,
          borderRadius: 999,
          backgroundColor: accent,
          opacity: 0.2,
          filter: "blur(100px)",
          transform: `translate(${orb1X}px, ${orb1Y}px)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 420,
          height: 420,
          borderRadius: 999,
          right: -100,
          top: 120,
          backgroundColor: THEME.accentEmerald,
          opacity: 0.18,
          filter: "blur(90px)",
          transform: `translate(${orb2X}px, ${orb2Y}px)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 360,
          height: 360,
          borderRadius: 999,
          left: -120,
          bottom: -120,
          backgroundColor: THEME.accentAmber,
          opacity: 0.16,
          filter: "blur(90px)",
          transform: `translate(${orb3X}px, ${orb3Y}px)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(148, 163, 184, 0.16) 1px, transparent 1px), linear-gradient(90deg, rgba(148, 163, 184, 0.12) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          opacity: 0.15,
          transform: `translateY(${gridShift}px)`,
        }}
      />
    </AbsoluteFill>
  );
};
const IntroScene: React.FC<{
  durationInFrames: number;
  brandName: string;
  website: string;
}> = ({ durationInFrames, brandName, website }) => {
  const codeLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.codeKeyword }}>const</span> course = {"{"}
    </>,
    <>
      &nbsp;&nbsp;focus: <span style={{ color: THEME.codeString }}>"LLM 训练"</span>,
    </>,
    <>
      &nbsp;&nbsp;modules: [
      <span style={{ color: THEME.codeString }}>"基础"</span>,
      <span style={{ color: THEME.codeString }}>"架构"</span>,
      <span style={{ color: THEME.codeString }}>"训练"</span>
      ],
    </>,
    <>
      &nbsp;&nbsp;experiments: <span style={{ color: THEME.codeValue }}>true</span>,
    </>,
    <>
      &nbsp;&nbsp;repo: <span style={{ color: THEME.codeString }}>"minimind.wiki"</span>,
    </>,
    <>{"}"};</>,
    <>
      <span style={{ color: THEME.codeKeyword }}>export default</span> course;
    </>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentBlue}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 560, display: "flex", flexDirection: "column", gap: 12 }}>
            <SiteBadge text={website} accent={THEME.accentBlue} />
            <div
              style={{
                fontSize: 16,
                letterSpacing: "0.35em",
                textTransform: "uppercase",
                color: THEME.muted,
              }}
            >
              {brandName}
            </div>
            <SceneTitle
              title="从零理解 LLM 训练"
              subtitle="拒绝黑箱，每个选择都讲清楚。"
            />
            <div style={{ fontSize: 18, color: THEME.muted }}>
              实验 + 代码 + 清晰路线图。
            </div>
            <TagRow
              accent={THEME.accentBlue}
              items={["模块化课程", "可运行实验", "训练脚本"]}
              startDelay={14}
            />
          </div>
        }
        right={
          <CodePanel
            title="intro.ts"
            lines={codeLines}
            accent={THEME.accentBlue}
            footer="// 边做边学"
          />
        }
      />
    </SceneShell>
  );
};

const FocusScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const codeLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.codeKeyword }}>const</span> ablation = compare({"{"}
    </>,
    <>
      &nbsp;&nbsp;baseline: <span style={{ color: THEME.codeString }}>"基线"</span>,
    </>,
    <>
      &nbsp;&nbsp;variantA: <span style={{ color: THEME.codeString }}>"归一化 + 残差"</span>,
    </>,
    <>
      &nbsp;&nbsp;variantB: <span style={{ color: THEME.codeString }}>"注意力改进"</span>,
    </>,
    <>{"}"});</>,
    <>
      report.plot(ablation);
    </>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentViolet}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 560 }}>
            <SceneTitle
              title="学会权衡取舍"
              subtitle="看清每个设计决策如何影响训练表现。"
            />
            <BulletList
              accent={THEME.accentViolet}
              items={[
                "并排对照实验，不靠猜测",
                "小数据集快速反馈",
                "先建立直觉，再扩展规模",
              ]}
            />
          </div>
        }
        right={<CodePanel title="ablation.ts" lines={codeLines} accent={THEME.accentViolet} />}
      />
    </SceneShell>
  );
};

const LearningPathScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cards = [
    {
      title: "01 基础篇",
      desc: "归一化、位置编码、注意力",
      accent: THEME.accentBlue,
    },
    {
      title: "02 架构篇",
      desc: "残差、Transformer 模块、扩展规律",
      accent: THEME.accentViolet,
    },
    {
      title: "03 训练篇",
      desc: "数据集、训练循环、评估、推理",
      accent: THEME.accentEmerald,
    },
  ];

  const codeLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.codeKeyword }}>const</span> modules = [
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeString }}>"基础"</span>,
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeString }}>"架构"</span>,
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeString }}>"训练"</span>,
    </>,
    <>{"]"};</>,
    <>
      roadmap.render(modules);
    </>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentViolet}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 740 }}>
            <SceneTitle
              title="模块化学习路径"
              subtitle="每个模块都配有可运行实验。"
            />
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 20,
                marginTop: 24,
              }}
            >
              {cards.map((card, index) => {
                const cardIn = spring({
                  frame: frame - (10 + index * 6),
                  fps,
                  config: { damping: 160, stiffness: 120 },
                });
                return (
                  <div
                    key={card.title}
                    style={{
                      padding: "20px",
                      borderRadius: 18,
                      background: "rgba(15, 23, 42, 0.65)",
                      border: "1px solid rgba(148, 163, 184, 0.2)",
                      boxShadow: "0 12px 30px rgba(15, 23, 42, 0.25)",
                      opacity: cardIn,
                      transform: `translateY(${interpolate(cardIn, [0, 1], [20, 0])}px)`,
                    }}
                  >
                    <div
                      style={{
                        width: 46,
                        height: 4,
                        borderRadius: 999,
                        background: card.accent,
                        marginBottom: 12,
                      }}
                    />
                    <div style={{ fontSize: 20, fontWeight: 600 }}>
                      {card.title}
                    </div>
                    <div
                      style={{
                        marginTop: 10,
                        color: THEME.muted,
                        fontSize: 16,
                        lineHeight: 1.4,
                      }}
                    >
                      {card.desc}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        }
        right={<CodePanel title="roadmap.ts" lines={codeLines} accent={THEME.accentViolet} />}
      />
    </SceneShell>
  );
};
const ExperimentsScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const bars = [
    { label: "基线", value: 0.46, color: THEME.accentBlue },
    { label: "变体 A", value: 0.72, color: THEME.accentEmerald },
    { label: "变体 B", value: 0.86, color: THEME.accentAmber },
  ];

  const chartHeight = 160;

  const terminalLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.accentEmerald }}>$</span> run --dataset tiny-shakespeare
    </>,
    <span style={{ color: THEME.muted }}>step 200 | loss 2.13 | ppl 8.4</span>,
    <span style={{ color: THEME.muted }}>变体A +0.18 | 变体B +0.31</span>,
    <span style={{ color: THEME.muted }}>保存检查点中...</span>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentEmerald}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 720 }}>
            <SceneTitle
              title="用实验获得清晰结论"
              subtitle="用测量结果说话，而不是凭经验传说。"
            />
            <div
              style={{
                marginTop: 18,
                padding: "20px",
                borderRadius: 20,
                background: "rgba(15, 23, 42, 0.7)",
                border: "1px solid rgba(148, 163, 184, 0.2)",
              }}
            >
              <div
                style={{
                  height: chartHeight,
                  display: "flex",
                  alignItems: "flex-end",
                  gap: 20,
                }}
              >
                {bars.map((bar, index) => {
                  const barIn = spring({
                    frame: frame - (10 + index * 6),
                    fps,
                    config: { damping: 160, stiffness: 120 },
                  });
                  const height = interpolate(barIn, [0, 1], [0, bar.value * chartHeight]);
                  return (
                    <div
                      key={bar.label}
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 10,
                      }}
                    >
                      <div
                        style={{
                          width: "100%",
                          height: chartHeight,
                          display: "flex",
                          alignItems: "flex-end",
                          background: "rgba(15, 23, 42, 0.4)",
                          borderRadius: 16,
                          padding: 8,
                          border: "1px solid rgba(148, 163, 184, 0.15)",
                        }}
                      >
                        <div
                          style={{
                            width: "100%",
                            height,
                            borderRadius: 12,
                            background: bar.color,
                            boxShadow: `0 12px 30px ${bar.color}40`,
                          }}
                        />
                      </div>
                      <div style={{ fontSize: 16, color: THEME.muted }}>{bar.label}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        }
        right={<TerminalPanel title="train.log" lines={terminalLines} accent={THEME.accentEmerald} />}
      />
    </SceneShell>
  );
};

const BuildScene: React.FC<{ durationInFrames: number }> = ({ durationInFrames }) => {
  const codeLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.codeKeyword }}>function</span> transformerBlock(x) {"{"}
    </>,
    <>
      &nbsp;&nbsp;const y = attention(x);
    </>,
    <>
      &nbsp;&nbsp;const z = mlp(norm(x + y));
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeKeyword }}>return</span> norm(x + y + z);
    </>,
    <>{"}"}</>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentBlue}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 720 }}>
            <SceneTitle
              title="搭建真实组件"
              subtitle="从分词器到训练循环。"
            />
            <CardRow
              items={[
                { title: "分词器", desc: "文本转 ID、词表、采样" },
                { title: "Transformer", desc: "模块、注意力、MLP" },
                { title: "训练器", desc: "损失、评估、检查点" },
              ]}
            />
          </div>
        }
        right={<CodePanel title="model.ts" lines={codeLines} accent={THEME.accentBlue} />}
      />
    </SceneShell>
  );
};

const AssetsScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const tree = [
    { name: "modules/", level: 0 },
    { name: "01-foundation/", level: 1 },
    { name: "02-architecture/", level: 1 },
    { name: "experiments/", level: 1 },
    { name: "learning_materials/", level: 0 },
    { name: "learning_log.md", level: 0 },
    { name: "knowledge_base.md", level: 0 },
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentAmber}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 720 }}>
            <SceneTitle
              title="代码与笔记保持同步"
              subtitle="每段讲解都对应可运行脚本。"
            />
            <CardRow
              items={[
                { title: "脚本", desc: "整洁且可复现实验" },
                { title: "教程", desc: "按步骤拆解模块" },
                { title: "结果", desc: "附带预期输出" },
              ]}
            />
          </div>
        }
        right={<FileTreePanel title="项目目录" items={tree} accent={THEME.accentAmber} />}
      />
    </SceneShell>
  );
};
const WorkflowScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const steps = [
    "克隆仓库",
    "运行小型实验",
    "对比结果",
    "扩展思路",
  ];

  const terminalLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.accentBlue }}>$</span> git clone minimind.wiki
    </>,
    <>
      <span style={{ color: THEME.accentBlue }}>$</span> pnpm install
    </>,
    <>
      <span style={{ color: THEME.accentBlue }}>$</span> pnpm run docs:dev
    </>,
    <span style={{ color: THEME.muted }}>服务已启动: http://localhost:5173</span>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentEmerald}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 720 }}>
            <SceneTitle
              title="快速反馈工作流"
              subtitle="短周期迭代，形成真实直觉。"
            />
            <StepRow items={steps} />
          </div>
        }
        right={<TerminalPanel title="终端" lines={terminalLines} accent={THEME.accentEmerald} />}
      />
    </SceneShell>
  );
};

const NotesScene: React.FC<{ durationInFrames: number }> = ({ durationInFrames }) => {
  const codeLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.codeKeyword }}>const</span> notes = {"{"}
    </>,
    <>
      &nbsp;&nbsp;learningLog: <span style={{ color: THEME.codeString }}>"每日洞察"</span>,
    </>,
    <>
      &nbsp;&nbsp;knowledgeBase: <span style={{ color: THEME.codeString }}>"概念 + 公式"</span>,
    </>,
    <>
      &nbsp;&nbsp;roadmap: <span style={{ color: THEME.codeString }}>"从基础到生产"</span>,
    </>,
    <>{"}"};</>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentViolet}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 720 }}>
            <SceneTitle
              title="持续生长的知识库"
              subtitle="记录你学到了什么，以及为什么重要。"
            />
            <CardRow
              items={[
                { title: "学习日志", desc: "每日洞察与实验" },
                { title: "知识库", desc: "概念、公式、权衡" },
                { title: "路线图", desc: "从基础到生产" },
              ]}
            />
          </div>
        }
        right={<CodePanel title="notes.ts" lines={codeLines} accent={THEME.accentViolet} />}
      />
    </SceneShell>
  );
};

const OutcomesScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const codeLines: React.ReactNode[] = [
    <>
      <span style={{ color: THEME.codeKeyword }}>const</span> takeaway = [
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeString }}>"复现实验"</span>,
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeString }}>"解释决策"</span>,
    </>,
    <>
      &nbsp;&nbsp;<span style={{ color: THEME.codeString }}>"自信扩展"</span>,
    </>,
    <>{"]"};</>,
  ];

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentBlue}>
      <SplitLayout
        left={
          <div style={{ maxWidth: 620 }}>
            <SceneTitle
              title="面试表达更有底气"
              subtitle="用证据解释权衡。"
            />
            <BulletList
              accent={THEME.accentBlue}
              items={[
                "复现核心实验",
                "清晰说明设计决策",
                "自信扩展到更大模型",
              ]}
            />
          </div>
        }
        right={<CodePanel title="takeaways.ts" lines={codeLines} accent={THEME.accentBlue} />}
      />
    </SceneShell>
  );
};

const CtaScene: React.FC<{
  durationInFrames: number;
  brandName: string;
  website: string;
}> = ({ durationInFrames, brandName, website }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleIn = spring({ frame, fps, config: SPRING_SMOOTH });
  const subtitleIn = spring({ frame: frame - 8, fps, config: SPRING_SMOOTH });
  const buttonIn = spring({ frame: frame - 16, fps, config: SPRING_SMOOTH });

  return (
    <SceneShell durationInFrames={durationInFrames} accent={THEME.accentBlue}>
      <div
        style={{
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          gap: 16,
        }}
      >
        <div
          style={{
            fontFamily: CODE_FONT,
            fontSize: 22,
            letterSpacing: "0.3em",
            color: THEME.accentBlue,
          }}
        >
          <GlitchText text={website} />
        </div>
        <div
          style={{
            fontSize: 52,
            fontWeight: 700,
            opacity: titleIn,
            transform: `translateY(${interpolate(titleIn, [0, 1], [16, 0])}px)`,
          }}
        >
          开始你的 LLM 训练之旅
        </div>
        <div
          style={{
            fontSize: 22,
            color: THEME.muted,
            opacity: subtitleIn,
            transform: `translateY(${interpolate(subtitleIn, [0, 1], [14, 0])}px)`,
          }}
        >
          {brandName} 让理论落地为实践
        </div>
        <div
          style={{
            marginTop: 8,
            padding: "12px 26px",
            borderRadius: 999,
            background: THEME.accentBlue,
            color: "#0b1020",
            fontWeight: 700,
            fontSize: 18,
            boxShadow: "0 16px 30px rgba(56, 189, 248, 0.3)",
            opacity: buttonIn,
            transform: `translateY(${interpolate(buttonIn, [0, 1], [12, 0])}px)`,
          }}
        >
          打开学习路线图
        </div>
        <div style={{ fontSize: 16, color: THEME.muted, marginTop: 6 }}>
          {website}
        </div>
      </div>
    </SceneShell>
  );
};
const SplitLayout: React.FC<{
  left: React.ReactNode;
  right: React.ReactNode;
  reverse?: boolean;
}> = ({ left, right, reverse = false }) => {
  const primary = reverse ? right : left;
  const secondary = reverse ? left : right;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: reverse ? "0.9fr 1.1fr" : "1.1fr 0.9fr",
        gap: 28,
        alignItems: "center",
      }}
    >
      <div>{primary}</div>
      <div>{secondary}</div>
    </div>
  );
};

const SiteBadge: React.FC<{ text: string; accent: string }> = ({
  text,
  accent,
}) => {
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        padding: "6px 12px",
        borderRadius: 999,
        border: `1px solid ${accent}88`,
        background: `${accent}22`,
        fontFamily: CODE_FONT,
        fontSize: 13,
        letterSpacing: "0.25em",
        textTransform: "uppercase",
        color: THEME.text,
        width: "fit-content",
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: 999,
          background: accent,
          boxShadow: `0 0 10px ${accent}`,
        }}
      />
      {text}
    </div>
  );
};

const CodingOverlay: React.FC<{
  durationInFrames: number;
  accent: string;
}> = ({ durationInFrames, accent }) => {
  const frame = useCurrentFrame();

  const driftSlow = interpolate(frame, [0, durationInFrames], [10, -24], {
    extrapolateRight: "clamp",
  });
  const driftFast = interpolate(frame, [0, durationInFrames], [-12, 28], {
    extrapolateRight: "clamp",
  });
  const scanShift = interpolate(frame, [0, durationInFrames], [-30, 30], {
    extrapolateRight: "clamp",
  });
  const flicker = interpolate(frame % 30, [0, 15, 30], [0.12, 0.05, 0.12]);

  const codeA = [
    "const run = async () => {",
    "  const data = load(\"tiny-shakespeare\");",
    "  const model = buildTransformer(cfg);",
    "  for (step of steps) train(model, data);",
    "  saveCheckpoint(model);",
    "};",
  ];
  const codeB = [
    "git clone https://minimind.wiki",
    "pnpm install",
    "pnpm run docs:dev",
    "python modules/01-foundation/...",
    "compare --baseline --variants",
  ];

  return (
    <AbsoluteFill style={{ pointerEvents: "none", zIndex: 1 }}>
      <CodeStream
        lines={codeA}
        accent={THEME.accentBlue}
        offset={driftSlow}
        style={{ top: 70, right: 70, width: 320, opacity: 0.12 }}
      />
      <CodeStream
        lines={codeB}
        accent={accent}
        offset={driftFast}
        style={{ bottom: 60, left: 70, width: 300, opacity: 0.1 }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px)",
          backgroundSize: "100% 6px",
          opacity: 0.15,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `linear-gradient(180deg, transparent 0%, ${accent}15 50%, transparent 100%)`,
          opacity: flicker,
          transform: `translateY(${scanShift}px)`,
        }}
      />
      <CornerMarks accent={accent} />
    </AbsoluteFill>
  );
};

const CodeStream: React.FC<{
  lines: string[];
  accent: string;
  offset: number;
  style?: React.CSSProperties;
}> = ({ lines, accent, offset, style }) => {
  return (
    <div
      style={{
        position: "absolute",
        fontFamily: CODE_FONT,
        fontSize: 12,
        lineHeight: 1.6,
        color: accent,
        transform: `translateY(${offset}px)`,
        ...style,
      }}
    >
      {lines.map((line, index) => (
        <div key={`${line}-${index}`}>{line}</div>
      ))}
    </div>
  );
};

const CornerMarks: React.FC<{ accent: string }> = ({ accent }) => {
  const style = {
    position: "absolute" as const,
    width: 28,
    height: 28,
    borderColor: `${accent}66`,
  };

  return (
    <>
      <div
        style={{
          ...style,
          top: 26,
          left: 26,
          borderTop: `1px solid ${accent}66`,
          borderLeft: `1px solid ${accent}66`,
        }}
      />
      <div
        style={{
          ...style,
          top: 26,
          right: 26,
          borderTop: `1px solid ${accent}66`,
          borderRight: `1px solid ${accent}66`,
        }}
      />
      <div
        style={{
          ...style,
          bottom: 26,
          left: 26,
          borderBottom: `1px solid ${accent}66`,
          borderLeft: `1px solid ${accent}66`,
        }}
      />
      <div
        style={{
          ...style,
          bottom: 26,
          right: 26,
          borderBottom: `1px solid ${accent}66`,
          borderRight: `1px solid ${accent}66`,
        }}
      />
    </>
  );
};
const CodePanel: React.FC<{
  title: string;
  lines: React.ReactNode[];
  accent: string;
  footer?: string;
}> = ({ title, lines, accent, footer }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const panelIn = spring({
    frame: frame - 6,
    fps,
    config: { damping: 160, stiffness: 120 },
  });

  return (
    <div
      style={{
        background: THEME.panel,
        border: `1px solid ${THEME.panelBorder}`,
        borderRadius: 16,
        padding: 16,
        fontFamily: CODE_FONT,
        fontSize: 13,
        lineHeight: 1.6,
        color: "rgba(226, 232, 240, 0.9)",
        boxShadow: "0 20px 40px rgba(5, 10, 20, 0.45)",
        opacity: panelIn,
        transform: `translateY(${interpolate(panelIn, [0, 1], [24, 0])}px)`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <div style={{ display: "flex", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 999, background: "#f87171" }} />
          <span style={{ width: 10, height: 10, borderRadius: 999, background: "#fbbf24" }} />
          <span style={{ width: 10, height: 10, borderRadius: 999, background: "#34d399" }} />
        </div>
        <div style={{ fontSize: 12, color: THEME.muted }}>{title}</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {lines.map((line, index) => (
          <div key={`line-${index}`} style={{ display: "flex", gap: 12 }}>
            <div style={{ width: 24, color: THEME.codeComment }}>
              {String(index + 1).padStart(2, "0")}
            </div>
            <div style={{ flex: 1 }}>{line}</div>
          </div>
        ))}
      </div>
      {footer ? (
        <div style={{ marginTop: 10, color: accent, fontSize: 12 }}>{footer}</div>
      ) : null}
    </div>
  );
};

const TerminalPanel: React.FC<{
  title: string;
  lines: React.ReactNode[];
  accent: string;
}> = ({ title, lines, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const panelIn = spring({
    frame: frame - 6,
    fps,
    config: { damping: 160, stiffness: 120 },
  });
  const cursorOn = Math.floor(frame / 12) % 2 === 0;

  return (
    <div
      style={{
        background: "rgba(8, 12, 22, 0.85)",
        border: `1px solid ${THEME.panelBorder}`,
        borderRadius: 16,
        padding: 16,
        fontFamily: CODE_FONT,
        fontSize: 13,
        lineHeight: 1.6,
        color: "rgba(226, 232, 240, 0.9)",
        boxShadow: "0 20px 40px rgba(5, 10, 20, 0.45)",
        opacity: panelIn,
        transform: `translateY(${interpolate(panelIn, [0, 1], [24, 0])}px)`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <div style={{ fontSize: 12, color: THEME.muted }}>{title}</div>
        <div style={{ color: accent }}>实时</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {lines.map((line, index) => (
          <div key={`terminal-${index}`}>{line}</div>
        ))}
        <div style={{ display: "flex", gap: 6, color: accent }}>
          <span>$</span>
          <span style={{ opacity: cursorOn ? 1 : 0 }}>█</span>
        </div>
      </div>
    </div>
  );
};

const FileTreePanel: React.FC<{
  title: string;
  items: { name: string; level: number }[];
  accent: string;
}> = ({ title, items, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const panelIn = spring({
    frame: frame - 6,
    fps,
    config: { damping: 160, stiffness: 120 },
  });

  return (
    <div
      style={{
        background: THEME.panel,
        border: `1px solid ${THEME.panelBorder}`,
        borderRadius: 16,
        padding: 16,
        fontFamily: CODE_FONT,
        fontSize: 13,
        lineHeight: 1.6,
        color: "rgba(226, 232, 240, 0.9)",
        boxShadow: "0 20px 40px rgba(5, 10, 20, 0.45)",
        opacity: panelIn,
        transform: `translateY(${interpolate(panelIn, [0, 1], [24, 0])}px)`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <div style={{ fontSize: 12, color: THEME.muted }}>{title}</div>
        <div style={{ color: accent }}>目录</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {items.map((item, index) => (
          <div
            key={`tree-${index}`}
            style={{
              paddingLeft: item.level * 12,
              color: item.name.endsWith("/") ? accent : THEME.text,
            }}
          >
            {item.name}
          </div>
        ))}
      </div>
    </div>
  );
};

const GlitchText: React.FC<{ text: string }> = ({ text }) => {
  const frame = useCurrentFrame();
  const shift = Math.sin(frame / 3) * 2;

  return (
    <span style={{ position: "relative", display: "inline-block" }}>
      <span
        style={{
          position: "absolute",
          left: -2,
          top: 0,
          color: THEME.accentViolet,
          opacity: 0.6,
          transform: `translateX(${shift}px)`,
        }}
      >
        {text}
      </span>
      <span
        style={{
          position: "absolute",
          left: 2,
          top: 0,
          color: THEME.accentBlue,
          opacity: 0.6,
          transform: `translateX(${-shift}px)`,
        }}
      >
        {text}
      </span>
      <span style={{ position: "relative" }}>{text}</span>
    </span>
  );
};

const SceneTitle: React.FC<{
  title: string;
  subtitle?: string;
}> = ({ title, subtitle }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleIn = spring({ frame, fps, config: SPRING_SMOOTH });
  const subtitleIn = spring({ frame: frame - 6, fps, config: SPRING_SMOOTH });

  return (
    <div style={{ maxWidth: 980 }}>
      <div
        style={{
          fontSize: 50,
          fontWeight: 700,
          lineHeight: 1.08,
          opacity: titleIn,
          transform: `translateY(${interpolate(titleIn, [0, 1], [16, 0])}px)`,
        }}
      >
        {title}
      </div>
      {subtitle ? (
        <div
          style={{
            fontSize: 22,
            fontWeight: 500,
            color: THEME.muted,
            marginTop: 10,
            opacity: subtitleIn,
            transform: `translateY(${interpolate(subtitleIn, [0, 1], [14, 0])}px)`,
          }}
        >
          {subtitle}
        </div>
      ) : null}
    </div>
  );
};

const TagRow: React.FC<{
  items: string[];
  accent: string;
  startDelay: number;
}> = ({ items, accent, startDelay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 16 }}>
      {items.map((item, index) => {
        const tagIn = spring({
          frame: frame - (startDelay + index * 6),
          fps,
          config: SPRING_SMOOTH,
        });
        return (
          <div
            key={item}
            style={{
              padding: "8px 14px",
              borderRadius: 999,
              border: `1px solid ${accent}66`,
              background: `${accent}22`,
              fontSize: 16,
              opacity: tagIn,
              transform: `translateY(${interpolate(tagIn, [0, 1], [12, 0])}px)`,
            }}
          >
            {item}
          </div>
        );
      })}
    </div>
  );
};

const BulletList: React.FC<{
  items: string[];
  accent: string;
}> = ({ items, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 18 }}>
      {items.map((item, index) => {
        const bulletIn = spring({
          frame: frame - (8 + index * 6),
          fps,
          config: SPRING_SMOOTH,
        });
        return (
          <div
            key={item}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              fontSize: 20,
              color: THEME.text,
              opacity: bulletIn,
              transform: `translateY(${interpolate(bulletIn, [0, 1], [12, 0])}px)`,
            }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 999,
                background: accent,
                boxShadow: `0 0 12px ${accent}99`,
              }}
            />
            {item}
          </div>
        );
      })}
    </div>
  );
};

const CardRow: React.FC<{
  items: { title: string; desc: string }[];
}> = ({ items }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: 20,
        marginTop: 22,
      }}
    >
      {items.map((item, index) => {
        const cardIn = spring({
          frame: frame - (10 + index * 6),
          fps,
          config: { damping: 160, stiffness: 120 },
        });
        return (
          <div
            key={item.title}
            style={{
              padding: "18px",
              borderRadius: 16,
              background: "rgba(15, 23, 42, 0.65)",
              border: "1px solid rgba(148, 163, 184, 0.2)",
              opacity: cardIn,
              transform: `translateY(${interpolate(cardIn, [0, 1], [18, 0])}px)`,
            }}
          >
            <div style={{ fontSize: 20, fontWeight: 600 }}>{item.title}</div>
            <div style={{ marginTop: 10, color: THEME.muted, fontSize: 16 }}>
              {item.desc}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const StepRow: React.FC<{ items: string[] }> = ({ items }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 16,
        marginTop: 20,
      }}
    >
      {items.map((item, index) => {
        const stepIn = spring({
          frame: frame - (8 + index * 5),
          fps,
          config: { damping: 160, stiffness: 120 },
        });
        return (
          <div
            key={item}
            style={{
              padding: "16px",
              borderRadius: 14,
              background: "rgba(15, 23, 42, 0.65)",
              border: "1px solid rgba(148, 163, 184, 0.2)",
              opacity: stepIn,
              transform: `translateY(${interpolate(stepIn, [0, 1], [16, 0])}px)`,
            }}
          >
            <div
              style={{
                width: 30,
                height: 30,
                borderRadius: 10,
                background: "rgba(56, 189, 248, 0.2)",
                color: THEME.accentBlue,
                fontWeight: 700,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: 10,
              }}
            >
              {index + 1}
            </div>
            <div style={{ fontSize: 16, color: THEME.text }}>{item}</div>
          </div>
        );
      })}
    </div>
  );
};
