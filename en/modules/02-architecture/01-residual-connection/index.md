---
title: Residual Connection | MiniMind LLM Training
description: Learn how residual connections preserve information, improve gradient flow, and make deep Transformers easier to optimize.
keywords: residual connection, gradient flow, Transformer, deep network
---

# 01. Residual Connection

> Learn a delta while keeping a direct path for information and gradients.

## Learning goals

After this module, you will be able to:

- Explain the difference between `y = F(x)` and `y = x + F(x)`
- Use Jacobians to explain improved gradient flow
- Distinguish an identity path from a guarantee of perfect stability
- Locate both residual paths in MiniMind
- Verify training, gradient-flow, and depth-scaling behavior with experiments

## Learning path

1. Read the [teaching notes](./teaching.md)
2. Run the three comparison experiments
3. Read the [MiniMind code walkthrough](./code_guide.md)
4. Complete the [quiz](./quiz.md)

```bash
cd modules/02-architecture/01-residual-connection/experiments
bash run_all.sh
```

## Experiments

| Experiment | Question | Measurement |
|---|---|---|
| `exp1_with_vs_without.py` | Is a near-identity task easier with residuals? | Final MSE at equal depth |
| `exp2_gradient_flow.py` | Can gradients reach early layers? | Input/output gradient ratio |
| `exp3_depth_scaling.py` | Is input information preserved as depth grows? | Cosine similarity to input |

The scripts write JSON results and contain deterministic assertions. A failed conclusion returns a non-zero exit code.

## Core formula

Plain sub-layer:

$$
y = F(x), \qquad \frac{\partial y}{\partial x} = J_F
$$

Residual sub-layer:

$$
y = x + F(x), \qquad \frac{\partial y}{\partial x} = I + J_F
$$

The identity path contributes the direct term $I$, so gradients do not depend entirely on the residual branch.

## In MiniMind

```python
residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)
hidden_states += residual

hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

Each `MiniMindBlock` has one residual path around Attention and another around the MLP.

## Completion check

- [ ] Write the forward and backward residual formulas
- [ ] Explain why a near-zero residual branch can be a useful starting point
- [ ] Explain why both branches need matching shapes
- [ ] Identify both residual paths in `MiniMindBlock.forward`
- [ ] Pass `bash experiments/run_all.sh`

## Next

Continue to [02. Transformer Block](../02-transformer-block/).
