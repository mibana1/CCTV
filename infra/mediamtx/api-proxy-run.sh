#!/bin/sh
set -eu

nginx -g 'daemon off;' &
nginx_pid=$!

while kill -0 "$nginx_pid" 2>/dev/null; do
  if ! wget -q --spider --timeout=2 http://127.0.0.1:9997/v3/config/global/get; then
    kill "$nginx_pid" 2>/dev/null || true
    wait "$nginx_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done

wait "$nginx_pid"
