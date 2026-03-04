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

if [[ "${SSH_LINES[$LINE_INDEX]}" = "true" ]]; then
    echo "SSH config line $((LINE_INDEX + 1)) in $SSH_FILE is set to 'true'" >&2
    exit 0
fi

if [[ "${SSH_LINES[$LINE_INDEX]}" = "false" ]]; then
    echo "SSH config line $((LINE_INDEX + 1)) in $SSH_FILE is set to 'false'" >&2
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
POD_NAME="grce_pod$((LINE_INDEX + 1))"

join_cmd() {
    local IFS=" "
    echo "$*"
}

run_ssh() {
    "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
}

open_shell() {
    "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST"
}

pod_init() {
        run_ssh "set -ex; ln -sf /usr/share/zoneinfo/Europe/Vienna /etc/localtime; mkdir -p $REMOTE_DIR/data $REMOTE_DIR/model; echo $POD_NAME > /.podname; apt update; apt install -y rsync tmux; pip install --break-system-packages tokenizers transformers nvidia-ml-py"
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/rpwd.py" "${REMOTE_HOST}:/root/rpwd.py"
        run_ssh "python3 /root/rpwd.py"
}

rsync_update() {
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/grce.py" "${REMOTE_HOST}:${REMOTE_DIR}/"
}

rsync_put() {
    local dataset_filter="${1:-}"
    if [[ ! -d "$ROOT_DIR/data" ]]; then
        echo "Warning: $ROOT_DIR/data directory not found; skipping." >&2
        return
    fi
    if [[ -z "$dataset_filter" ]]; then
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/data/" "${REMOTE_HOST}:${REMOTE_DIR}/data/"
        return
    fi
    shopt -s nullglob
    local matches=("$ROOT_DIR/data/${dataset_filter}_"*)
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "Warning: No files matched data/${dataset_filter}_*; nothing uploaded." >&2
        return
    fi
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "${matches[@]}" "${REMOTE_HOST}:${REMOTE_DIR}/data/"
}

rsync_push() {
    if [[ -d "$LOCAL_MODEL_DIR" ]]; then
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$LOCAL_MODEL_DIR/" "${REMOTE_HOST}:${REMOTE_DIR}/model/"
    else
        echo "Warning: $LOCAL_MODEL_DIR directory not found; skipping." >&2
    fi
}

rsync_pull() {
    mkdir -p "$LOCAL_MODEL_DIR"
    local rc=0
    if ! rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "${REMOTE_HOST}:${REMOTE_DIR}/model/" "$LOCAL_MODEL_DIR/"; then
        rc=$?
        if [[ $rc -eq 23 ]]; then
            echo "Warning: rsync reported partial transfer (code 23); retrying once..." >&2
            if ! rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "${REMOTE_HOST}:${REMOTE_DIR}/model/" "$LOCAL_MODEL_DIR/"; then
                rc=$?
                if [[ $rc -eq 23 ]]; then
                    echo "Warning: rsync still reports code 23; continuing despite partial transfer." >&2
                    return 0
                fi
                return $rc
            fi
        else
            return $rc
        fi
    fi
}

rsync_peek() {
    mkdir -p "$LOCAL_MODEL_DIR"
    rsync "${RSYNC_COMMON[@]}" \
        --exclude='*.pt' -e "$(join_cmd "${RSYNC_SSH[@]}")" \
        "${REMOTE_HOST}:${REMOTE_DIR}/model/" "$LOCAL_MODEL_DIR/"
}

case "${1:-}" in
    go)
        pod_init
        rsync_update
	open_shell
        ;;
    init)
        pod_init
        ;;
    shell)
	open_shell
        ;;
    update)
        rsync_update
        ;;
    put)
        shift
        rsync_put "$@"
        ;;
    push)
        rsync_push
        ;;
    pull)
        rsync_pull
        ;;
    peek)
        rsync_peek
        ;;
    *)
        echo "Usage: bash pod.sh [CFG] {go|init|shell|update|put|push|pull}" >&2
        exit 1
        ;;
esac
