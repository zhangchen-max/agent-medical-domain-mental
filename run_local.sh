#!/usr/bin/env bash
# 一键运行 Psych Debate Agent（本地 qwen3-32B-FP16）
#
# 用法:
#   1. 先在一个终端启动 vLLM: ./scripts/start_vllm.sh
#   2. 在另一个终端运行: ./run_local.sh
#
# 或使用 tmux/screen 在后台启动 vLLM 后直接运行本脚本

cd "$(dirname "$0")"
exec ./scripts/run.sh
