"""Prepare/push a private temporary GPU encoder and securely connect local API."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import requests
from dotenv import load_dotenv
from common import atomic_json
from manage import HERE, ROOT, WORK, authenticate, state


def prepare():
    output = WORK / 'encoder_notebook'
    output.mkdir(parents=True,exist_ok=True)
    identity = WORK/'encoder_identity.json'
    if identity.exists():
        kernel=json.loads(identity.read_text())['kernel']
    else:
        try: kernel=state()['kernel']+'-encoder'
        except RuntimeError:
            _,username=authenticate()
            kernel=f'{username}/image-retrieval-gpu-{uuid.uuid4().hex[:8]}'
        atomic_json(identity,{'kernel':kernel})
    code = "import pathlib, subprocess, sys\nCODE=pathlib.Path('/tmp/image-query-encoder')\nCODE.mkdir(exist_ok=True)\n"
    for name in ('common.py','pipeline.py','reranker.py','encoder_service.py','config.json','requirements-encoder.lock.txt'):
        code += f"(CODE/{name!r}).write_text({(HERE/name).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
    release = (HERE/'cloudflared_release.json').read_text()
    code += f"(CODE/'cloudflared_release.json').write_text({release!r})\n"
    code += """subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.7'],check=True)
subprocess.run([sys.executable,'-m','uv','venv','--python','3.12','/tmp/image-query-env'],check=True)
PYTHON='/tmp/image-query-env/bin/python'
subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'-r',str(CODE/'requirements-encoder.lock.txt')],check=True)
subprocess.run([PYTHON,'-u',str(CODE/'encoder_service.py')],check=True)
"""
    atomic_json(output/'encoder.ipynb', {'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'id':'query-encoder','cell_type':'code','execution_count':None,'outputs':[],
                  'metadata':{},'source':code.splitlines(keepends=True)}]})
    atomic_json(output/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],
        'code_file':'encoder.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[],'kernel_sources':[],'competition_sources':[],'model_sources':[]})
    return output,kernel


def connect(kernel, minutes=25):
    load_dotenv(ROOT/'.env',override=True)
    deadline=time.monotonic()+minutes*60
    last_count=0
    while time.monotonic()<deadline:
        try:
            with requests.get('https://api.kaggle.com/v1/kernels/logs/stream/'+kernel,
                headers={'Authorization':'Bearer '+os.environ['KAGGLE_API_TOKEN'],'Accept':'text/event-stream'},
                timeout=(15,20),stream=True) as response:
                response.raise_for_status()
                events=[]
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith('data:'): continue
                    payload=line[5:].strip()
                    if payload=='END_OF_LOG': break
                    event=json.loads(payload)
                    text=event.get('data','')
                    events.append(text)
                    marker='IMAGE_ENCODER_CONNECTION:'
                    if marker in text:
                        connection=json.JSONDecoder().raw_decode(text.split(marker,1)[1].lstrip())[0]
                        health=requests.get(connection['url']+'/health',headers={'Authorization':'Bearer '+connection['key']},timeout=(15,60))
                        health.raise_for_status()
                        local=json.loads((HERE/'config.json').read_text())['embedding']
                        if health.json().get('embedding_config')!=local: raise RuntimeError('Encoder config mismatch')
                        env=ROOT/'.env'
                        lines=env.read_text(encoding='utf-8').splitlines()
                        from reranker import MODEL, REVISION
                        if health.json().get('reranker_revision')!=REVISION or health.json().get('reranker_model')!=MODEL:
                            raise RuntimeError('Reranker service version mismatch; push the updated notebook')
                        changes={'IMAGE_ENCODER_URL':connection['url'],'IMAGE_ENCODER_KEY':connection['key'],'IMAGE_ENCODER_MODE':'remote','IMAGE_RERANKER_MODE':'remote'}
                        lines=[line for line in lines if line.split('=',1)[0] not in changes]
                        env.write_text('\n'.join(lines+[k+'='+v for k,v in changes.items()])+'\n',encoding='utf-8')
                        atomic_json(WORK/'encoder_status.json',{'url':connection['url'],'expires_at_utc':connection['expires_at_utc'],'ready':True})
                        print('GPU encoder connected; credentials saved in local .env. Restart local API.',flush=True)
                        return
                if len(events)>last_count:
                    print(f'Encoder startup log events: {len(events)}',flush=True)
                    last_count=len(events)
        except requests.exceptions.RequestException:
            print('Waiting for encoder startup / API connection',flush=True)
        time.sleep(20)
    raise TimeoutError('Encoder connection deadline reached; inspect private Kaggle notebook')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','push','connect'])
    args=parser.parse_args()
    try:
        directory,kernel=prepare()
        if args.command=='push':
            api,_=authenticate()
            response=api.kernels_push(str(directory),timeout='10800',acc='NvidiaTeslaT4')
            if response.error: raise RuntimeError(response.error)
            print('Private GPU encoder submitted: https://www.kaggle.com/code/'+kernel,flush=True)
        if args.command=='connect': connect(kernel)
    except Exception as exc:
        # No exception text: signed URLs and connection records may contain keys.
        print('Encoder command failed:',type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
