#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_FILE="$ROOT_DIR/.podssh"
REMOTE_DIR="/workspace/grce"

if [[ ! -f "$SSH_FILE" ]]; then
    echo "Missing SSH config at $SSH_FILE" >&2
    exit 1
fi

readarray -t SSH_LINES < "$SSH_FILE"
if [[ ${#SSH_LINES[@]} -eq 0 ]]; then
    echo "Invalid SSH config in $SSH_FILE" >&2
    exit 1
fi

LINE_INDEX=0
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    LINE_INDEX=$(( ${1} - 1 ))
    shift
fi
if [[ $LINE_INDEX -lt 0 || $LINE_INDEX -ge ${#SSH_LINES[@]} ]]; then
    echo "Requested SSH config line $((LINE_INDEX + 1)) not found in $SSH_FILE" >&2
    exit 1
fi

read -r -a SSH_ARGS <<< "${SSH_LINES[$LINE_INDEX]}"
if [[ "${#SSH_ARGS[@]}" -lt 2 ]]; then
    echo "Invalid SSH config line $((LINE_INDEX + 1)) in $SSH_FILE" >&2
    exit 1
fi

SSH_BIN="${SSH_ARGS[0]}"
REMOTE_HOST="${SSH_ARGS[1]}"
SSH_OPTS=("${SSH_ARGS[@]:2}")
SSH_OPTS+=("-o" "StrictHostKeyChecking=no" "-o" "UserKnownHostsFile=/dev/null")
RSYNC_SSH=("$SSH_BIN" "${SSH_OPTS[@]}")
RSYNC_COMMON=(-avz --no-perms --no-owner --no-group)
LOCAL_MODEL_DIR="$ROOT_DIR/model_pod$((LINE_INDEX + 1))"

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
    # ensure_remote_dirs
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/grce.py" "${REMOTE_HOST}:${REMOTE_DIR}/"
}

rsync_put() {
    ensure_remote_dirs
    rsync_update
    if [[ -d "$ROOT_DIR/data" ]]; then
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/data/" "${REMOTE_HOST}:${REMOTE_DIR}/data/"
    else
        echo "Warning: $ROOT_DIR/data directory not found; skipping." >&2
    fi
}

rsync_push() {
    ensure_remote_dirs
    rsync_update
    if [[ -d "$LOCAL_MODEL_DIR" ]]; then
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$LOCAL_MODEL_DIR/" "${REMOTE_HOST}:${REMOTE_DIR}/model/"
    else
        echo "Warning: $LOCAL_MODEL_DIR directory not found; skipping." >&2
    fi
}

rsync_pull() {
    mkdir -p "$LOCAL_MODEL_DIR"
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "${REMOTE_HOST}:${REMOTE_DIR}/model/" "$LOCAL_MODEL_DIR/"
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
    put)
        rsync_put
        ;;
    push)
        rsync_push
        ;;
    pull)
        rsync_pull
        ;;
    shell)
        ensure_remote_dirs
        open_shell
        ;;
    *)
        echo "Usage: bash pod.sh [CFG] {init|go|update|put|push|pull|shell}" >&2
        exit 1
        ;;
esac
