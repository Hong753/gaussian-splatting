"""Exercise dependency reuse and failure paths without compiling CUDA packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('scenario', ['install', 'cached', 'shadowed', 'missing'])
def test_setup_reuses_existing_knn_and_builds_only_plane_rasterizer(tmp_path, scenario):
    root=Path(__file__).resolve().parents[1]
    scripts=tmp_path/'project/scripts';scripts.mkdir(parents=True)
    shutil.copyfile(root/'scripts/setup_gsprior.sh',scripts/'setup_gsprior.sh')
    repo=tmp_path/'GSPrior';repo.mkdir()
    marker=repo/'orange4f_adapter.json';marker.write_text('{"upstream":"fixture"}')
    env=tmp_path/'venv';(env/'bin').mkdir(parents=True)
    base=tmp_path/'base_packages';base.mkdir()
    (base/'torch.py').write_text('__version__="fixture"\nclass cuda:\n    is_available=staticmethod(lambda: True)\n')
    (base/'diff_plane_rasterization.py').write_text('')
    (base/'pycuda').mkdir();(base/'pycuda/__init__.py').write_text('');(base/'pycuda/driver.py').write_text('')
    def package(directory):
        (directory/'simple_knn').mkdir(parents=True)
        (directory/'simple_knn/__init__.py').write_text('')
        (directory/'simple_knn/_C.py').write_text('def distCUDA2(points): return points\n')
    if scenario!='missing': package(base)
    local=tmp_path/'venv_packages';local.mkdir()
    if scenario=='shadowed': package(local)
    if scenario=='cached': shutil.copyfile(marker,env/'orange4f_ready')
    # Run the installer's real inline Python checks against controlled package
    # locations. Only pip and checkout patching are substituted and recorded.
    wrapper='#!'+sys.executable+'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ["CALL_LOG"]).open("a") as out:
    out.write(json.dumps(args)+"\\n")
sys.path.insert(0,os.environ["BASE_PACKAGES"])
if Path(sys.argv[0]).parent.parent == Path(os.environ["GSPRIOR_ENV"]):
    sys.path.insert(0,os.environ["LOCAL_PACKAGES"])
if args[:2]==["-m","pip"]:
    raise SystemExit(0)
if args and args[0]=="scripts/setup_gsprior.py":
    raise SystemExit(0)
if args and args[0]=="-":
    sys.argv=["-",*args[1:]]
    exec(compile(sys.stdin.read(),"<installer stdin>","exec"))
elif args and args[0]=="-c":
    sys.argv=["-c",*args[2:]]
    exec(compile(args[1],"<installer command>","exec"))
else:
    raise SystemExit("Unexpected invocation: "+repr(args))
'''
    bindir=tmp_path/'bin';bindir.mkdir()
    for path in (bindir/'python',env/'bin/python'):
        path.write_text(wrapper);path.chmod(0o755)
    log=tmp_path/'calls.jsonl'
    result=subprocess.run(['bash',str(scripts/'setup_gsprior.sh')],
        env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
             'GSPRIOR_REPO':str(repo),'GSPRIOR_ENV':str(env),'BASE_PACKAGES':str(base),
             'LOCAL_PACKAGES':str(local),'CALL_LOG':str(log)},
        text=True,capture_output=True,timeout=30)
    calls=[json.loads(row) for row in log.read_text().splitlines()]
    installs=[args for args in calls if args[:2]==['-m','pip']]
    assert not any('simple-knn' in arg or 'simple_knn' in arg for args in installs for arg in args)
    if scenario in ('shadowed','missing'):
        assert result.returncode!=0 and not installs
        assert ('shadows' if scenario=='shadowed' else 'working 3dgs environment') in result.stderr
        assert not (env/'orange4f_ready').exists()
    else:
        assert result.returncode==0,result.stdout+result.stderr
        assert f'Reusing 3dgs simple_knn: {base}/simple_knn/_C.py' in result.stdout
        assert (env/'orange4f_ready').read_bytes()==marker.read_bytes()
        builds=[args for args in installs if '--no-build-isolation' in args]
        if scenario=='install':
            assert len(builds)==1 and builds[0][-1]==str(repo/'submodules/diff-plane-rasterization')
        else:
            assert not installs
