# Residual Connection 实验结果

运行：

```bash
cd modules/02-architecture/01-residual-connection/experiments
bash run_all.sh
```

脚本会生成：

- `exp1_with_vs_without.json`：有/无残差的训练损失对比
- `exp2_gradient_flow.json`：输入端与输出端梯度范数对比
- `exp3_depth_scaling.json`：不同深度下的信息保留对比

所有实验使用固定随机种子并包含断言。任一核心结论不成立时，脚本会返回非零退出码。
