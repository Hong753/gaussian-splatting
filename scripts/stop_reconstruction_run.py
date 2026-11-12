#!/usr/bin/env python3
"""Stop this user's exact named run, including its trainer and queued evaluator."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

SCRIPTS = {'run_fixed_lidar.py','run_distance_weighted.py','run_soft_surface_long.py',
           'train.py','render.py','render_views_worker.py','eval_all_frames.py'}


def matches(argv,model,cwd):
    if not argv or not Path(argv[0]).name.startswith('python'):
        return False
    args = list(argv[1:])
    while args and args[0] in ('-u','-B','-O','-OO','-s','-S','-E','-I','-q'):
        args.pop(0)
    if not args or Path(args[0]).name not in SCRIPTS:
        return False
    for index,arg in enumerate(args):
        value = None
        if arg in ('-m','--model','--model_path') and index+1 < len(args):
            value = args[index+1]
        elif arg.startswith(('--model=','--model_path=')):
            value = arg.split('=',1)[1]
        if value is not None:
            path = Path(value).expanduser()
            if (path if path.is_absolute() else cwd/path).resolve() == model:
                return True
    return False


def candidates(model,proc=Path('/proc'),uid=None):
    model=Path(model).resolve(); uid=os.getuid() if uid is None else uid
    result=[]
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0] == 'Z':
                continue
            argv=[os.fsdecode(v) for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
            if matches(argv,model,(directory/'cwd').resolve()):
                result.append(dict(pid=int(directory.name),start=fields[19],argv=argv))
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--stop',action='store_true')
    args=p.parse_args(argv)
    model=args.model.expanduser().resolve()
    plan=json.loads((model/'periodic_evaluation/plan.json').read_text())
    if Path(plan['model']).resolve() != model or plan['gpu'] != args.gpu:
        raise ValueError('Named run does not match the requested GPU/model; no processes signaled')
    rows=candidates(model)
    for row in rows:
        print(f'GPU {args.gpu} run, PID {row["pid"]}: {" ".join(row["argv"])}',flush=True)
    if not args.stop:
        return 0
    for row in rows:
        try:
            fd=os.pidfd_open(row['pid'])
            try:
                current={r['pid']:r for r in candidates(model)}.get(row['pid'])
                if current is not None and current['start'] == row['start']:
                    signal.pidfd_send_signal(fd,signal.SIGTERM)
            finally:
                os.close(fd)
        except ProcessLookupError:
            continue
    for _ in range(100):
        remaining=candidates(model)
        if not remaining:
            print(f'No trainer/evaluator remains for {model}. Saved checkpoints/results retained.',flush=True)
            return 0
        time.sleep(.2)
    raise RuntimeError(f'Processes still active; replacement was not started: {[r["pid"] for r in remaining]}')


if __name__ == '__main__':
    raise SystemExit(main())
