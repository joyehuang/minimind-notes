#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:---full}"

case "$MODE" in
  --quick)
    ;;
  --full)
    ;;
  *)
    echo "用法: bash run_all.sh [--full|--quick]"
    exit 2
    ;;
esac

echo "运行 Residual Connection 模块完整实验测试"
echo "Python: $PYTHON_BIN"
echo "模式: $MODE"

run_experiment() {
  if [ "$MODE" = "--quick" ]; then
    "$PYTHON_BIN" "$1" --quick
  else
    "$PYTHON_BIN" "$1"
  fi
}

run_experiment exp1_with_vs_without.py
run_experiment exp2_gradient_flow.py
run_experiment exp3_depth_scaling.py

echo "Residual Connection 模块全部实验通过"
