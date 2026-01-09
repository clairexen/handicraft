#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_FILE="$ROOT_DIR/.podssh"
REMOTE_DIR="/workspace/grce"

if [[ ! -f "$SSH_FILE" ]]; then
    echo "Missing SSH config at $SSH_FILE" >&2
    exit 1
fi

read -r -a SSH_ARGS < "$SSH_FILE"
if [[ "${#SSH_ARGS[@]}" -lt 2 ]]; then
    echo "Invalid SSH config in $SSH_FILE" >&2
    exit 1
fi

SSH_BIN="${SSH_ARGS[0]}"
REMOTE_HOST="${SSH_ARGS[1]}"
SSH_OPTS=("${SSH_ARGS[@]:2}")
RSYNC_SSH=("$SSH_BIN" "${SSH_OPTS[@]}")
RSYNC_COMMON=(-avz --no-perms --no-owner --no-group)

join_cmd() {
    local IFS=" "
    echo "$*"
}

run_ssh() {
    "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
}

ensure_remote_dirs() {
    run_ssh "mkdir -p $REMOTE_DIR $REMOTE_DIR/model"
}

rsync_update() {
    ensure_remote_dirs
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/grce.py" "${REMOTE_HOST}:${REMOTE_DIR}/"
}

rsync_push() {
    ensure_remote_dirs
    rsync_update
    if [[ -d "$ROOT_DIR/model" ]]; then
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/model/" "${REMOTE_HOST}:${REMOTE_DIR}/model/"
    else
        echo "Warning: $ROOT_DIR/model directory not found; skipping." >&2
    fi
}

rsync_pull() {
    mkdir -p "$ROOT_DIR/model"
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "${REMOTE_HOST}:${REMOTE_DIR}/model/" "$ROOT_DIR/model/"
}

open_shell() {
    "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST"
}

case "${1:-}" in
    init)
        ensure_remote_dirs
        rsync_update
        ;;
    go)
        ensure_remote_dirs
        run_ssh "apt update && apt install -y rsync"
        run_ssh "cd $REMOTE_DIR && pip install tokenizers transformers"
        rsync_update
        open_shell
        ;;
    update)
        rsync_update
        ;;
    push)
        rsync_push
        ;;
    pull)
        rsync_pull
        ;;
    *)
        echo "Usage: bash pod.sh {init|go|update|push|pull}" >&2
        exit 1
        ;;
esac
