---
title: Transformer Block | MiniMind LLM Training
description: Assemble RMSNorm, Attention, FeedForward, and residual connections into a complete MiniMind Transformer block.
keywords: Transformer block, decoder block, Pre-Norm, MiniMind architecture
---

# 02. Transformer Block

> Individual components solve local problems; the block defines how they cooperate.

## Learning goals

After this module, you will be able to:

- Implement a Pre-Norm decoder block from scratch
- Explain the roles of Attention, FFN, Norm, and double residuals
- Explain why `Attention → FFN` is canonical without treating it as the only valid order
- Understand how position embeddings, attention masks, and KV Cache enter a block
- Stack blocks and verify shape, causality, and gradients

## Learning path

1. Read the [teaching notes](./teaching.md)
2. Run the three assembly experiments
3. Read the [MiniMind code walkthrough](./code_guide.md)
4. Complete the [quiz](./quiz.md)

```bash
cd modules/02-architecture/02-transformer-block/experiments
bash run_all.sh
```

## Experiments

| Experiment | Question | Verification |
|---|---|---|
| `exp1_component_ablation.py` | What do Mixer, FFN, and residuals contribute? | Compare final MSE |
| `exp2_order_matters.py` | Is swapping components equivalent? | Measure output difference |
| `exp3_stack_and_causality.py` | Can a complete decoder block be safely stacked? | Verify shape, causality, and gradients |

## Core structure

```python
h = x + attention(attention_norm(x))
out = h + feed_forward(ffn_norm(h))
```

## Completion check

- [ ] Implement a Pre-Norm Transformer block
- [ ] Explain why the block has two norms and two residual paths
- [ ] Explain why component order changes the function
- [ ] Explain why KV Cache is independent per layer
- [ ] Pass `bash experiments/run_all.sh`

## Next

Return to the [Architecture overview](../) and continue to the full model and training pipeline.
