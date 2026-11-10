#!/usr/bin/env python3
"""Stop only this user's named reconstruction render jobs on selected GPUs."""
import argparse
import csv
import io
import os
from pathlib import Path
import signal
import subprocess
import time


def gpu_processes(gpu_csv, process_csv, selected):
    uuids = {row[1].strip(): int(row[0]) for row in csv.reader(io.StringIO(gpu_csv)) if row}
    processes = {}
    for row in csv.reader(io.StringIO(process_csv)):
        if row and row[0].strip() in uuids:
            processes.setdefault(int(row[1]), set()).add(uuids[row[0].strip()])
    return {pid: gpus for pid, gpus in processes.items() if gpus and gpus <= set(selected)}


def matches_job(argv, runs):
    scripts = {'render.py', 'render_views_worker.py', 'render_ablation_videos.py', 'eval_all_frames.py'}
    return (any(Path(arg).name in scripts for arg in argv) and
            any(run in Path(arg).parts for run in runs for arg in argv))


def candidates(processes, runs, proc=Path('/proc'), uid=None):
    uid = os.getuid() if uid is None else uid
    result = []
    for pid, gpus in processes.items():
        directory = proc/str(pid)
        try:
            if directory.stat().st_uid != uid:
                continue
            argv = [a.decode() for a in (directory/'cmdline').read_bytes().split(b'\0') if a]
            # Start time binds the decision to this process, preventing PID reuse.
            start = (directory/'stat').read_text().rsplit(')', 1)[1].split()[19]
            if matches_job(argv, runs):
                result.append(dict(pid=pid, gpus=sorted(gpus), argv=argv, start=start))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--runs', nargs='+', default=['proxygs_kf600_original', 'proxygs_stride3_original'])
    parser.add_argument('--stop', action='store_true', help='Send SIGTERM; otherwise list matching jobs only')
    args = parser.parse_args(argv)
    def query(fields):
        return subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader,nounits'], text=True)
    rows = candidates(gpu_processes(query('--query-gpu=index,uuid'), query('--query-compute-apps=gpu_uuid,pid'), args.gpus), args.runs)
    for row in rows:
        print(f"GPU {row['gpus']}, PID {row['pid']}: {' '.join(row['argv'])}", flush=True)
    if not rows:
        print('No matching render jobs on the requested GPUs.')
        return 0
    if not args.stop:
        return 0
    signaled = []
    for row in rows:
        pid = row['pid']
        try:
            # pidfd guarantees the signal cannot hit a recycled PID.
            descriptor = os.pidfd_open(pid)
            try:
                current = candidates({pid: set(row['gpus'])}, args.runs)
                if not current or current[0]['start'] != row['start']:
                    continue
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                signaled.append(pid)
            finally:
                os.close(descriptor)
        except ProcessLookupError:
            continue
    for _ in range(25):
        remaining = candidates({r['pid']: set(r['gpus']) for r in rows if r['pid'] in signaled}, args.runs)
        if not remaining:
            print(f'Stopped {len(signaled)} matching render jobs.')
            return 0
        time.sleep(.2)
    print(f'SIGTERM sent; these jobs still appear active: {[r["pid"] for r in remaining]}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
