#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:---full}"

case "$MODE" in
  --quick|--full)
    ;;
  *)
    echo "用法: bash run_all.sh [--full|--quick]"
    exit 2
    ;;
esac

run_experiment() {
  if [ "$MODE" = "--quick" ]; then
    "$PYTHON_BIN" "$1" --quick
  else
    "$PYTHON_BIN" "$1"
  fi
}

echo "运行 Transformer Block 模块完整实验测试"
echo "Python: $PYTHON_BIN"
echo "模式: $MODE"

run_experiment exp1_component_ablation.py
run_experiment exp2_order_matters.py
run_experiment exp3_stack_and_causality.py

echo "Transformer Block 模块全部实验通过"
