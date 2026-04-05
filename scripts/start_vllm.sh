#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务，加载 qwen3-32B-FP16
# 需先安装: pip install vllm

set -e
cd "$(dirname "$0")/.."

# 从 .env 读取模型路径，若无则使用默认
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
MODEL_PATH="${LOCAL_MODEL_PATH:-/mnt/data/zhangchen/models/qwen3-32B-FP16}"
SERVED_NAME="${LOCAL_MODEL:-qwen3-32B-FP16}"
PORT="${VLLM_PORT:-8000}"

# 使用 GPU 0-3
[ -n "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 解决 libstdc++ CXXABI_1.3.15 缺失
if [ -n "$CONDA_PREFIX" ]; then
  CONDA_ROOT="$(dirname "$(dirname "$CONDA_PREFIX")")"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "启动 vLLM，模型: $MODEL_PATH"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0,1,2,3} | Tensor Parallel: $TP_SIZE 卡"
echo "OpenAI API 地址: http://localhost:$PORT/v1"
echo "按 Ctrl+C 停止服务"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code
