#!/usr/bin/env bash
# 创建 conda 环境 psy-debate，与项目 .venv311 等效

set -e
cd "$(dirname "$0")/.."

echo "创建 conda 环境 psy-debate (Python 3.11)..."
conda env create -f environment.yml

echo ""
echo "激活环境并安装本项目..."
eval "$(conda shell.bash hook)"
conda activate psy-debate
pip install -e .

echo ""
echo "完成。使用方式:"
echo "  conda activate psy-debate"
echo "  ./scripts/start_vllm.sh    # 终端1"
echo "  ./run_local.sh             # 终端2"
