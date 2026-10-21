#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/lu/miniforge3/envs/dermaguard/bin/python"

cd "$PROJECT_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "错误：未找到 DermaGuard Python 环境：$PYTHON" >&2
  exit 1
fi

# stdout is reserved for the MCP JSON-RPC protocol. Human-readable status
# messages must go to stderr so MCP clients can still use this script directly.
echo "霁肤 · Jiderma MCP 服务器已启动（stdio 模式）。" >&2
echo "当前终端会等待 MCP 客户端请求；按 Ctrl+C 停止。" >&2
echo "若要打开网页，请另行运行：bash scripts/run_mock_demo.sh" >&2
exec "$PYTHON" -m backend.app.mcp.server
