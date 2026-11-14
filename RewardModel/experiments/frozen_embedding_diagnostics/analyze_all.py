"""Run disjoint representation cases on the four permitted devices."""
import os
from pathlib import Path
import subprocess
import sys

from study import OUT,GPUS,write_json


def run():
    script=str(Path(__file__).with_name('analysis.py'))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
    subprocess.run([sys.executable,script,'--prepare'],env=env,check=True)
    workers=[]
    for worker,gpu in enumerate(GPUS):
        log=(OUT/'logs'/f'analysis_worker{worker}.log').open('a')
        child=subprocess.Popen([sys.executable,script,'--gpu',str(gpu),'--worker',str(worker),'--workers','4'],
            env={**env,'CUDA_VISIBLE_DEVICES':GPUS[gpu]},stdout=log,stderr=subprocess.STDOUT)
        workers.append((child,log,worker,gpu))
    write_json(OUT/'analysis_workers.json',[dict(pid=p.pid,worker=w,gpu=g) for p,f,w,g in workers])
    results=[dict(worker=w,exit_code=p.wait()) for p,f,w,g in workers]
    for p,f,w,g in workers:f.close()
    write_json(OUT/'analysis_exit.json',results)
    if any(r['exit_code'] for r in results):raise RuntimeError('Analysis worker failed')
    subprocess.run([sys.executable,script,'--finalize'],env=env,check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],env={**env,'CUDA_VISIBLE_DEVICES':GPUS[0]},check=True)


if __name__=='__main__':run()
