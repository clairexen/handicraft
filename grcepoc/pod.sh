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
        run_ssh "set -ex; mkdir -p $REMOTE_DIR/data $REMOTE_DIR/model; echo $POD_NAME > $REMOTE_DIR/.podname; apt update; apt install -y rsync; pip install tokenizers transformers"
}

rsync_update() {
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/grce.py" "${REMOTE_HOST}:${REMOTE_DIR}/"
}

rsync_put() {
    rsync_update
    if [[ -d "$ROOT_DIR/data" ]]; then
        rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "$ROOT_DIR/data/" "${REMOTE_HOST}:${REMOTE_DIR}/data/"
    else
        echo "Warning: $ROOT_DIR/data directory not found; skipping." >&2
    fi
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
    rsync "${RSYNC_COMMON[@]}" -e "$(join_cmd "${RSYNC_SSH[@]}")" "${REMOTE_HOST}:${REMOTE_DIR}/model/" "$LOCAL_MODEL_DIR/"
}

case "${1:-}" in
    go)
        pod_init
        rsync_put
        rsync_push
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
        rsync_put
        ;;
    push)
        rsync_push
        ;;
    pull)
        rsync_pull
        ;;
    monitor)
        tail -n 20 -f model_pod*/.monitor.*.ansi
        ;;
    loop)
        while true; do
		echo
		echo "----------------------------"
		date
		echo "----------------------------"
		for pod_dir in model_pod[0-9]*; do
			echo; ( set -ex; bash pod.sh "${pod_dir#model_pod}" pull; )
			for ansi_file in $pod_dir/*.ansi; do
				sed -re 's/Running on remote pod/Monitoring remote pod/' \
					< $ansi_file > $pod_dir/.new_monitor.${ansi_file#$pod_dir/}
				new_monitor="$pod_dir/.new_monitor.${ansi_file#$pod_dir/}"
				monitor="$pod_dir/.monitor.${ansi_file#$pod_dir/}"
				# Keep .monitor files growing by only appending the new suffix
				if [[ -f "$monitor" ]]; then
					monitor_size=$(wc -c < "$monitor")
					if cmp -n "$monitor_size" "$monitor" "$new_monitor" >/dev/null 2>&1; then
						tail -c "+$((monitor_size + 1))" "$new_monitor" >> "$monitor"
					else
						cp "$new_monitor" "$monitor"
					fi
				else
					cp "$new_monitor" "$monitor"
				fi
				rm "$new_monitor"
			done
		done
		echo; ( set -ex; sleep 300; )
	done
        ;;
    *)
        echo "Usage: bash pod.sh [CFG] {go|init|shell|update|put|push|pull|monitor|loop}" >&2
        exit 1
        ;;
esac
