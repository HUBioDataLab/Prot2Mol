"""One bounded extraction controller; prior experiment queues are never invoked."""
import os
from pathlib import Path
import subprocess
import sys
import time

from study import OUT,GPUS,ROOT,write_json


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    write_json(OUT/'extraction_controller.json',dict(pid=os.getpid(),started=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()))
    script=str(Path(__file__).with_name('study.py'))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
    subprocess.run([sys.executable,script,'prepare'],env=env,check=True)
    workers=[]
    for side in ('query','candidate'):
        f=(logs/(side+'_download.log')).open('a')
        workers.append((subprocess.Popen([sys.executable,script,'download','--side',side],env=env,stdout=f,stderr=subprocess.STDOUT),f))
    failures=[p.wait() for p,f in workers]
    for p,f in workers:f.close()
    if any(failures):raise RuntimeError('Download failed: '+str(failures))
    workers=[]
    for side,state,gpu in [('query','pretrained',0),('candidate','pretrained',2),('query','random',5),('candidate','random',6)]:
        f=(logs/(side+'_'+state+'.log')).open('a')
        child_env={**env,'CUDA_VISIBLE_DEVICES':GPUS[gpu]}
        p=subprocess.Popen([sys.executable,script,'extract','--side',side,'--state',state,'--gpu',str(gpu)],env=child_env,stdout=f,stderr=subprocess.STDOUT)
        workers.append((p,f,side,state,gpu))
    write_json(OUT/'extraction_workers.json',[dict(pid=p.pid,side=s,state=t,gpu=g) for p,f,s,t,g in workers])
    results=[dict(side=s,state=t,exit_code=p.wait()) for p,f,s,t,g in workers]
    for p,f,s,t,g in workers:f.close()
    write_json(OUT/'extraction_exit.json',dict(workers=results,finished=time.time()))
    if any(r['exit_code'] for r in results):raise RuntimeError('Inspect failed extraction logs')


if __name__=='__main__':run()
