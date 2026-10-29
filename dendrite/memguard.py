#!/usr/bin/env python3
"""Host-memory watchdog for heavy runs on GB10. Runs a command, polls MemAvailable, and SIGKILLs the command's whole
process tree (explicit pids, never a process group) when it drops below --floor GiB.

Why: on GB10 CPU and GPU share one pool. gpucap.py caps what torch may allocate, but not what the rest of the box
needs; a 70 GiB GPU cap on a 61 GiB always-on floor sent the kernel OOM-killer into the user services (2026-10-26).
This guard keys on the quantity that actually runs out.

  memguard.py [--floor 8] [--tag NAME] -- cmd args...     exit = cmd's rc, or 137 if the guard killed it
  memguard.py --selftest                                  both directions; exit 0 = guard works
"""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path

PROGRESS = Path(os.environ.get("DENDRITE_PROGRESS", "progress.md"))   # the liveness log, in the working directory


def avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def tree(pid):
    """pid plus every descendant, read from /proc/<pid>/task/*/children."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop(); out.append(p)
        for t in Path(f"/proc/{p}/task").glob("*/children") if Path(f"/proc/{p}").exists() else []:
            try:
                todo += [int(c) for c in t.read_text().split()]
            except OSError:
                pass
    return out


def live(pid):
    """True for a running process; a killed child awaiting reaping by init is a zombie, not a survivor."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def log(msg):
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} memguard: {msg}\n")


def run(cmd, floor, tag, interval=0.5, grace=0.0):
    p = subprocess.Popen(cmd)
    low = 1e9
    time.sleep(grace)
    while p.poll() is None:
        a = avail_gib(); low = min(low, a)
        if a < floor:
            pids = tree(p.pid)
            for q in reversed(pids):
                try:
                    os.kill(q, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait()
            time.sleep(0.5)
            alive = [q for q in pids if live(q)]
            log(f"{tag} KILLED avail={a:.1f}GiB < floor={floor}GiB pids={pids} survivors={alive}")
            return (137 if not alive else 1), low
        time.sleep(interval)
    return p.returncode, low


def selftest():
    ok = True
    rc, _ = run(["sleep", "1"], floor=1, tag="selftest-allow")
    print(f"allow: benign cmd rc={rc} (want 0)"); ok &= rc == 0
    t = time.time()
    rc, _ = run(["sleep", "30"], floor=avail_gib() + 1, tag="selftest-deny")
    print(f"deny: floor above avail rc={rc} after {time.time() - t:.1f}s (want 137, <5s)"); ok &= rc == 137 and time.time() - t < 5
    rc, _ = run(["bash", "-c", "sleep 31 & sleep 31; wait"], floor=avail_gib() + 1, tag="selftest-tree", grace=1.0)
    left = subprocess.run(["pgrep", "-fx", "sleep 31"], capture_output=True, text=True).stdout.split()
    print(f"tree: nested children rc={rc}, survivors={left} (want 137, none)"); ok &= rc == 137 and not left
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=8.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    rc, low = run(cmd, a.floor, a.tag)
    log(f"{a.tag} END rc={rc} min_avail={low:.1f}GiB")
    sys.exit(rc)


if __name__ == "__main__":
    main()
