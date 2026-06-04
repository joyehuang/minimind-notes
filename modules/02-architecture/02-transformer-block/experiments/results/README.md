# Transformer Block 实验结果

运行：

```bash
cd modules/02-architecture/02-transformer-block/experiments
bash run_all.sh
```

脚本会生成：

- `exp1_component_ablation.json`：完整 Block 与组件消融的损失对比
- `exp2_order_matters.json`：交换组件顺序后的输出差异
- `exp3_stack_and_causality.json`：多层堆叠的形状、因果性与梯度

所有实验使用固定随机种子并包含断言。任一核心结论不成立时，脚本会返回非零退出码。
