#!/usr/bin/env bash
# 运行 Psych Debate Agent（本地 qwen3-32B-FP16）
# 使用前需先启动 vLLM: ./scripts/start_vllm.sh

set -e
cd "$(dirname "$0")/.."

# 检查 vLLM 服务是否可用
BASE_URL="${LOCAL_MODEL_BASE_URL:-http://localhost:8000/v1}"
HEALTH_URL="${BASE_URL%/v1}/health"
if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
  echo "vLLM 服务已就绪: $BASE_URL"
else
  echo "错误: 未检测到 vLLM 服务"
  echo "请先在另一个终端运行: ./scripts/start_vllm.sh"
  echo "或执行: python -m vllm.entrypoints.openai.api_server --model /mnt/data/zhangchen/models/qwen3-32B-FP16"
  exit 1
fi

# 加载 .env
[ -f .env ] && export $(grep -v '^#' .env | xargs)

# 激活环境：优先 conda psy-debate，否则 venv
if conda info --envs 2>/dev/null | grep -q "psy-debate "; then
  eval "$(conda shell.bash hook 2>/dev/null)" && conda activate psy-debate 2>/dev/null
elif [ -d .venv311 ] && [ -x .venv311/bin/python ]; then
  source .venv311/bin/activate
elif [ -d .venv ] && [ -x .venv/bin/python ]; then
  source .venv/bin/activate
fi

# 优先使用已安装的 psy-debate，否则用 python -m
if command -v psy-debate &>/dev/null; then
  exec psy-debate
else
  exec python -m psy_debate.main
fi
