"""Prepare/push a private temporary GPU encoder and securely connect local API."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

from dotenv import load_dotenv
from common import atomic_json
from manage import HERE, ROOT, WORK, authenticate
from model_registry import WORKERS, fingerprint


def credentials(slot):
    load_dotenv(ROOT / '.env', override=True)
    suffix = '' if slot == 1 else f'_{slot}'
    return os.environ.get(f'KAGGLE_USERNAME{suffix}'), os.environ.get(f'KAGGLE_API_TOKEN{suffix}')


def prepare(slot, group, username, resume=None):
    if not username or not re.fullmatch(r'[a-z0-9_-]+', username):
        raise ValueError(f'Set KAGGLE_USERNAME for slot {slot}')
    output = WORK / f'encoder_notebook_{slot}_{group}'
    output.mkdir(parents=True, exist_ok=True)
    identity = output / 'identity.json'
    if identity.exists():
        kernel = json.loads(identity.read_text(encoding='utf-8'))['kernel']
        if kernel.split('/')[0] != username:
            raise ValueError('Notebook identity belongs to a different account')
    else:
        kernel = f'{username}/image-retrieval-{group}-{uuid.uuid4().hex[:8]}'
        atomic_json(identity, {'kernel': kernel})
    code = "import os, pathlib, subprocess, sys\nCODE=pathlib.Path('/tmp/image-query-encoder')\nCODE.mkdir(exist_ok=True)\n"
    for name in ('common.py','pipeline.py','reranker.py','worker_service.py','model_registry.py','embed_corpus.py','config.json','requirements-encoder.lock.txt'):
        code += f"(CODE/{name!r}).write_text({(HERE/name).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
    code += f"os.environ['QWEN_RERANK_BATCH_SIZE']={'2' if group == 'qwen8b' else '8'!r}\n"
    release = (HERE/'cloudflared_release.json').read_text()
    code += f"(CODE/'cloudflared_release.json').write_text({release!r})\n"
    code += """subprocess.run([sys.executable,'-m','pip','install','--quiet','uv==0.9.7'],check=True)
subprocess.run([sys.executable,'-m','uv','venv','--python','3.12','/tmp/image-query-env'],check=True)
PYTHON='/tmp/image-query-env/bin/python'
subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'-r',str(CODE/'requirements-encoder.lock.txt'),'bitsandbytes==0.49.2','open-clip-torch==3.3.0'],check=True)
"""
    if group == 'index':
        code += f'RESUME={resume!r}\n'
        code += """from concurrent.futures import ThreadPoolExecutor
import shutil
sources = [p for p in pathlib.Path('/kaggle/input').rglob('images.parquet')
           if p.parent.name == 'artifacts' and (p.parent.parent / 'data').is_dir()]
if len(sources) != 1:
    raise RuntimeError('Expected a single mounted Castle5000 bundle with data/ and artifacts/')
artifacts = sources[0].parent
data = artifacts.parent / 'data'
output = pathlib.Path('/kaggle/working/embeddings')
if RESUME:
    checkpoints = list(pathlib.Path('/kaggle/input').rglob('embeddings'))
    if len(checkpoints) != 1:
        raise RuntimeError('Expected a single attached embeddings output')
    shutil.copytree(checkpoints[0], output, dirs_exist_ok=True)
def run(names, gpu):
    for name in names:
        subprocess.run([PYTHON, '-u', str(CODE/'embed_corpus.py'), '--artifacts', str(artifacts),
                        '--data', str(data), '--output', str(output), '--model', name,
                        '--gpu', str(gpu)], check=True)
with ThreadPoolExecutor(2) as pool:
    first = pool.submit(run, ('qwen8b', 'siglip2'), 0)
    second = pool.submit(run, ('qwen2b', 'eva', 'dfn'), 1)
    first.result()
    second.result()
"""
    else:
        code += f"subprocess.run([PYTHON,'-u',str(CODE/'worker_service.py'),'--group',{group!r}],check=True)\n"
    atomic_json(output/'encoder.ipynb', {'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'id':'query-encoder','cell_type':'code','execution_count':None,'outputs':[],
                  'metadata':{},'source':code.splitlines(keepends=True)}]})
    atomic_json(output/'kernel-metadata.json',{'id':kernel,'title':kernel.split('/')[1],
        'code_file':'encoder.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':['quands/image-retrieval-kitchen-5k-cc3d917a'] if group == 'index' else [],
        'kernel_sources':[resume] if resume else [], 'competition_sources':[], 'model_sources':[]})
    return output,kernel


def connect(kernel, slot, group, token, minutes=25):
    api, _ = authenticate()
    deadline=time.monotonic()+minutes*60
    last_count=0
    while time.monotonic()<deadline:
        try:
            events=0
            owner, slug = kernel.split('/', 1)
            with api.build_kaggle_client() as client:
                http=client._http_client
                http._init_session()
                http._session.trust_env=False
                url=f'{http._endpoint}/v1/kernels/logs/stream/{owner}/{slug}'
                headers=dict(http._session.headers)
                headers['Accept']='text/event-stream, */*'
                headers.pop('Content-Type', None)
                with http._session.get(url,stream=True,headers=headers,auth=http._session.auth,
                                       timeout=(15,20)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith('data:'): continue
                        payload=line[5:].lstrip()
                        if payload=='END_OF_LOG': break
                        try: text=json.loads(payload).get('data','')
                        except json.JSONDecodeError: text=payload
                        events+=1
                        marker='IMAGE_ENCODER_CONNECTION:'
                        if marker in text:
                            connection=json.JSONDecoder().raw_decode(text.split(marker,1)[1].lstrip())[0]
                            import requests
                            health_session=requests.Session()
                            health_session.trust_env=False
                            health = health_session.get(connection['url']+'/health',
                                headers={'Authorization': 'Bearer '+connection['key']}, timeout=(10, 20))
                            health.raise_for_status()
                            report = health.json()
                            expected = {name: fingerprint(name) for name in WORKERS[group]}
                            if report.get('group') != group or report.get('models') != expected or not report.get('ready'):
                                raise RuntimeError('Worker model configuration mismatch')
                            target = WORK / f'worker_{slot}_{group}.json'
                            atomic_json(target, {**connection, 'group': group})
                            print(f'Worker {slot} ({group}) connected.', flush=True)
                            return
            if events>last_count:
                print(f'Encoder startup log events: {events}',flush=True)
                last_count=events
        except Exception as exc:
            # Keep credentials and signed tunnel records out of output.
            print('Waiting for encoder startup / API connection ('+type(exc).__name__+')',flush=True)
        time.sleep(20)
    raise TimeoutError('Encoder connection deadline reached; inspect private Kaggle notebook')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','push','connect'])
    parser.add_argument('--workers', type=int, choices=(2, 3))
    parser.add_argument('--index', action='store_true', help='Prepare or push the corpus embedding notebook')
    parser.add_argument('--resume', help='Attach a previous Kaggle notebook output (owner/slug)')
    parser.add_argument('--slot', type=int, choices=(1, 2, 3), required=True)
    args=parser.parse_args()
    if args.index and args.command == 'connect':
        parser.error('Index jobs export artifacts instead of offering a live connection')
    if not args.index and args.workers is None:
        parser.error('Supply --workers for inference notebooks')
    if args.resume and (not args.index or not re.fullmatch(r'[a-z0-9_-]+/[a-z0-9_-]+', args.resume)):
        parser.error('--resume requires --index and a notebook owner/slug')
    groups = ('qwen8b', 'combined') if args.workers == 2 else ('qwen8b', 'qwen2b', 'small')
    if args.slot > len(groups):
        parser.error('--slot must be within --workers')
    group = 'index' if args.index else groups[args.slot-1]
    try:
        username, token = credentials(args.slot)
        directory, kernel = prepare(args.slot, group, username, args.resume)
        if args.command != 'prepare' and not token:
            parser.error(f'Missing Kaggle token for slot {args.slot}')
        if args.command=='push':
            os.environ['KAGGLE_API_TOKEN'] = token
            os.environ['KAGGLE_USERNAME'] = username
            from kaggle import api
            duration = '43200' if args.index else '10800'
            response=api.kernels_push(str(directory),timeout=duration,acc='NvidiaTeslaT4')
            if response.error: raise RuntimeError(response.error)
            print('Private GPU encoder submitted: https://www.kaggle.com/code/'+kernel,flush=True)
        if args.command == 'prepare': print(directory)
        if args.command=='connect': connect(kernel, args.slot, group, token)
    except Exception as exc:
        # No exception text: signed URLs and connection records may contain keys.
        print('Encoder command failed:',type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
