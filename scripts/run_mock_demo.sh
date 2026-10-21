#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="/home/lu/miniforge3/envs/dermaguard/bin"
BACKEND_LOG="$PROJECT_ROOT/artifacts/dermaguard_backend.log"
UI_LOG="$PROJECT_ROOT/artifacts/jiderma_react.log"

cd "$PROJECT_ROOT"
if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi
CHAT_PROVIDER="${DERMAGUARD_CHAT_PROVIDER:-mock}"
export DERMAGUARD_RAG_ADVANCED="${DERMAGUARD_RAG_ADVANCED:-1}"
case "$CHAT_PROVIDER" in
  deepseek) CHAT_MODEL="${DEEPSEEK_MODEL:-deepseek-flash}" ;;
  zhipu|zhipu_glm|glm) CHAT_MODEL="${ZHIPUAI_MODEL:-glm-4.7-flash}" ;;
  groq) CHAT_MODEL="${GROQ_MODEL:-未指定}" ;;
  *) CHAT_MODEL="本地模拟" ;;
esac
API_PORT="${DERMAGUARD_API_PORT:-8000}"
UI_PORT="${DERMAGUARD_UI_PORT:-8501}"
API_URL="http://127.0.0.1:$API_PORT"
UI_URL="http://127.0.0.1:$UI_PORT"

if [[ ! -x "$ENV_BIN/python" ]] || [[ ! -x "$ENV_BIN/node" ]]; then
  echo "错误：没有找到完整的 dermaguard 环境：$ENV_BIN"
  exit 1
fi
if [[ ! -f "$PROJECT_ROOT/frontend/node_modules/vite/bin/vite.js" ]]; then
  echo "错误：React 前端依赖尚未安装。"
  echo "请先执行：conda run -n dermaguard npm --prefix frontend install"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/artifacts"

port_has_http_service() {
  curl -sS --connect-timeout 0.2 --max-time 0.5 "http://127.0.0.1:$1/" \
    >/dev/null 2>&1
}

api_is_jiderma() {
  [[ "$(curl -fsS --connect-timeout 0.3 --max-time 1 "$API_URL/health" 2>/dev/null || true)" == *'"status":"ok"'* ]]
}

ui_is_jiderma() {
  curl -fsS --connect-timeout 0.3 --max-time 1 "$UI_URL/" 2>/dev/null \
    | grep -q '<title>霁肤 · Jiderma</title>'
}

if api_is_jiderma && ui_is_jiderma; then
  echo
  echo "霁肤 · Jiderma 已经在运行，无需重复启动："
  echo "  网页：$UI_URL"
  echo "  API 状态：$API_URL/health"
  if [[ "${DERMAGUARD_NO_BROWSER:-0}" != "1" ]] && command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$UI_URL" >/dev/null 2>&1 || true
  fi
  exit 0
fi

for port_name in "API:$API_PORT" "网页:$UI_PORT"; do
  service_name="${port_name%%:*}"
  port="${port_name##*:}"
  if port_has_http_service "$port"; then
    echo "错误：${service_name}端口 ${port} 已被占用，未启动新的服务。"
    echo "可先停止旧服务，或使用 DERMAGUARD_API_PORT/DERMAGUARD_UI_PORT 指定其他端口。"
    exit 1
  fi
done

cleanup() {
  trap - EXIT INT TERM
  for pid in "${UI_PID:-}" "${BACKEND_PID:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  echo
  echo "霁肤 · Jiderma 医生工作台已停止。"
}
trap cleanup EXIT INT TERM

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_BIN/uvicorn" backend.app.main:app --host 127.0.0.1 --port "$API_PORT" >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

ready=0
for _ in {1..30}; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    break
  fi
  if curl -fsS "$API_URL/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.3
done
if [[ "$ready" != "1" ]]; then
  echo "后端启动失败，请查看：$BACKEND_LOG"
  exit 1
fi

(cd "$PROJECT_ROOT/frontend" && \
  PATH="$ENV_BIN:$PATH" "$ENV_BIN/node" ./node_modules/vite/bin/vite.js \
    --host 127.0.0.1 --port "$UI_PORT") >"$UI_LOG" 2>&1 &
UI_PID=$!

ready=0
for _ in {1..30}; do
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    break
  fi
  if curl -fsS "$UI_URL/" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.3
done
if [[ "$ready" != "1" ]]; then
  echo "UI 启动失败，请查看：$UI_LOG"
  exit 1
fi

echo
echo "霁肤 · Jiderma 医生工作台已启动："
echo "  网页：$UI_URL"
echo "  API 文档：$API_URL/docs"
echo "  后端日志：$BACKEND_LOG"
echo "  UI 日志：$UI_LOG"
echo "  对话提供方：$CHAT_PROVIDER"
echo "  对话模型：$CHAT_MODEL"
echo
echo "按 Ctrl+C 可同时停止前后端。"

if [[ "${DERMAGUARD_NO_BROWSER:-0}" != "1" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$UI_URL" >/dev/null 2>&1 || true
fi

wait
