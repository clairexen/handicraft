#!/usr/bin/env python3
# RPWD -- Runpod Watchdog

import os
import sys
import time
import json
import signal
import subprocess
from datetime import datetime, timezone

SAMPLE_SECS = float(os.environ.get("SAMPLE_SECS", "5"))
MINUTES_IDLE = int(os.environ.get("MINUTES_IDLE", "10"))

CPU_UTIL_LIMIT = float(os.environ.get("CPU_UTIL_LIMIT", "8.0"))  # percent
GPU_UTIL_LIMIT = float(os.environ.get("GPU_UTIL_LIMIT", "5.0"))  # percent

LOG_PATH = os.environ.get("LOG_PATH", "/root/rpwd.log")
PIDFILE = os.environ.get("PIDFILE", "/tmp/rpwd.pid")

RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")  # optional; used for GraphQL fallback


def _read_proc_stat():
    # Returns (idle, total) jiffies
    with open("/proc/stat", "r", encoding="utf-8") as f:
        parts = f.readline().strip().split()
    if parts[0] != "cpu":
        raise RuntimeError("Unexpected /proc/stat format")
    vals = list(map(int, parts[1:]))
    # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals)
    return idle, total


def _cpu_util_over_interval(dt_sec: float) -> float:
    i1, t1 = _read_proc_stat()
    time.sleep(dt_sec)
    i2, t2 = _read_proc_stat()
    didle = i2 - i1
    dtotal = t2 - t1
    if dtotal <= 0:
        return 0.0
    util = (1.0 - (didle / dtotal)) * 100.0
    if util < 0:
        util = 0.0
    if util > 100.0:
        util = 100.0
    return util


def _mem_used_percent() -> float:
    # Uses MemAvailable to estimate actual used RAM.
    memtotal = None
    memavail = None
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                memtotal = int(line.split()[1])  # kB
            elif line.startswith("MemAvailable:"):
                memavail = int(line.split()[1])  # kB
            if memtotal is not None and memavail is not None:
                break
    if not memtotal or memavail is None:
        return 0.0
    used = memtotal - memavail
    return max(0.0, min(100.0, (used / memtotal) * 100.0))


def _query_gpus():
    # Returns list of dicts: [{"util": int, "mem_used": int, "mem_total": int}, ...]
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        raise RuntimeError(f"nvidia-smi failed: {e}")
    gpus = []
    if not out:
        return gpus
    for line in out.splitlines():
        # e.g. "12, 1024, 24576"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        util = int(float(parts[0]))
        mem_used = int(float(parts[1]))
        mem_total = int(float(parts[2])) if float(parts[2]) > 0 else 1
        gpus.append({"util": util, "mem_used": mem_used, "mem_total": mem_total})
    return gpus


def _stop_via_runpodctl(pod_id: str) -> bool:
    if not pod_id:
        return False
    try:
        subprocess.check_call(["runpodctl", "stop", "pod", pod_id])
        return True
    except Exception:
        return False


def _stop_via_graphql(pod_id: str, api_key: str) -> bool:
    if not pod_id or not api_key:
        return False
    query = {"query": f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}'}
    try:
        subprocess.check_call(
            ["curl", "-sS", "https://api.runpod.io/graphql",
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "Content-Type: application/json",
             "--data", json.dumps(query)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _log_line(line: str):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()


def _daemonize():
    # Double-fork daemonization
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if os.fork() > 0:
        os._exit(0)

    os.umask(0o027)
    os.chdir("/")

    # Redirect stdio to /dev/null
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "rb", 0) as f_in, open("/dev/null", "ab", 0) as f_out:
        os.dup2(f_in.fileno(), 0)
        os.dup2(f_out.fileno(), 1)
        os.dup2(f_out.fileno(), 2)

    # pidfile
    try:
        with open(PIDFILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()) + "\n")
    except Exception:
        pass


def main():
    _daemonize()
    idle_streak = 0

    _log_line(f"{datetime.now(timezone.utc).isoformat()}Z start pid={os.getpid()} "
              f"CPU_UTIL_LIMIT={CPU_UTIL_LIMIT} GPU_UTIL_LIMIT={GPU_UTIL_LIMIT} MINUTES_IDLE={MINUTES_IDLE}")

    while True:
        # Collect 60 seconds worth of samples
        n_samples = max(1, int(round(60.0 / SAMPLE_SECS)))
        cpu_utils = []
        mem_max = 0.0

        gpu_utils_sum = None
        gpu_utils_count = 0
        gpu_mem_max_pct = None  # list per GPU

        minute_start = time.time()
        for _ in range(n_samples):
            # CPU util sampled over SAMPLE_SECS (this sleep is inside _cpu_util_over_interval)
            cpu_u = _cpu_util_over_interval(SAMPLE_SECS)
            cpu_utils.append(cpu_u)

            mem_pct = _mem_used_percent()
            if mem_pct > mem_max:
                mem_max = mem_pct

            # GPU snapshot (no extra sleep)
            try:
                gpus = _query_gpus()
                if gpu_utils_sum is None:
                    gpu_utils_sum = [0.0] * len(gpus)
                    gpu_mem_max_pct = [0.0] * len(gpus)
                # If GPU count changes, re-init to current count
                if len(gpus) != len(gpu_utils_sum):
                    gpu_utils_sum = [0.0] * len(gpus)
                    gpu_mem_max_pct = [0.0] * len(gpus)
                    gpu_utils_count = 0

                for i, g in enumerate(gpus):
                    gpu_utils_sum[i] += float(g["util"])
                    mem_pct_gpu = (float(g["mem_used"]) / float(g["mem_total"])) * 100.0
                    if mem_pct_gpu > gpu_mem_max_pct[i]:
                        gpu_mem_max_pct[i] = mem_pct_gpu
                gpu_utils_count += 1
            except Exception as e:
                # Log once per minute later; keep going.
                pass

        # Normalize to exactly 1 minute boundaries (best-effort)
        elapsed = time.time() - minute_start
        if elapsed < 60.0:
            time.sleep(60.0 - elapsed)

        cpu_avg = sum(cpu_utils) / max(1, len(cpu_utils))

        if gpu_utils_sum is None or gpu_utils_count == 0:
            gpu_avg_list = []
            gpu_mem_max_list = []
            gpu_ok = False  # can't verify GPU idleness
        else:
            gpu_avg_list = [x / gpu_utils_count for x in gpu_utils_sum]
            gpu_mem_max_list = gpu_mem_max_pct
            gpu_ok = all(u < GPU_UTIL_LIMIT for u in gpu_avg_list)

        cpu_ok = cpu_avg < CPU_UTIL_LIMIT

        ts = datetime.now(timezone.utc).isoformat() + "Z"
        _log_line(
            f"{ts} "
            f"cpu_avg={cpu_avg:.2f}% mem_max={mem_max:.2f}% "
            f"gpu_avg={[round(u,2) for u in gpu_avg_list]} "
            f"gpu_mem_max={[round(m,2) for m in gpu_mem_max_list]} "
            f"idle={(cpu_ok and gpu_ok)} streak={idle_streak}"
        )

        if cpu_ok and gpu_ok:
            idle_streak += 1
        else:
            idle_streak = 0

        if idle_streak >= MINUTES_IDLE:
            _log_line(f"{ts} idle threshold met; attempting pod stop pod_id={RUNPOD_POD_ID!r}")
            ok = _stop_via_runpodctl(RUNPOD_POD_ID)
            if not ok:
                _log_line(f"{ts} runpodctl stop failed; trying GraphQL (requires RUNPOD_API_KEY)")
                ok = _stop_via_graphql(RUNPOD_POD_ID, RUNPOD_API_KEY)
            _log_line(f"{ts} stop_attempt_result={ok}")
            # Either way, exit; if the platform restarts the container, the watchdog will restart too.
            os._exit(0)


if __name__ == "__main__":
    main()
