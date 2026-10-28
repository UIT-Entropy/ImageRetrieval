"""Local CLI: prepare, upload private dataset/notebook, status, download outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from common import atomic_json, build_manifest, digest_file, fingerprint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / '.work'
STATE = WORK / 'state.json'


def authenticate():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    if not os.environ.get('KAGGLE_API_TOKEN'):
        raise RuntimeError('Set KAGGLE_API_TOKEN in the repository .env file')
    import kaggle
    api = kaggle.api
    username = api.get_config_value('username') or os.environ.get('KAGGLE_USERNAME')
    if not username:
        raise RuntimeError('Could not determine account. Set KAGGLE_USERNAME in .env')
    return api, username


def state():
    if not STATE.exists():
        raise RuntimeError('Run prepare first')
    return json.loads(STATE.read_text(encoding='utf-8'))


def scan_package(directory):
    """Strict file allowlist plus secret scan; .env is never copied."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    token = os.environ.get('KAGGLE_API_TOKEN', '')
    allowed = {'.jpg', '.json', '.jsonl', '.ipynb', '.txt', '.zip'}
    for path in Path(directory).rglob('*'):
        if not path.is_file():
            continue
        if path.suffix not in allowed or path.name.startswith('.env'):
            raise ValueError(f'Unexpected upload file: {path.name}')
        if path.suffix not in {'.jpg', '.zip'}:
            content = path.read_text(encoding='utf-8')
            if 'KGAT_' in content or (token and token in content) or 'hf_' in content and path.suffix != '.ipynb':
                raise ValueError(f'Possible credential in upload: {path.name}')
    return True


def notebook_sources():
    requirements = (HERE / 'requirements-kaggle.lock.txt').read_text(encoding='utf-8')
    bootstrap = '''import json, os, pathlib, subprocess, sys, traceback
BASE = pathlib.Path('/kaggle/working')
OUT = BASE / 'image_retrieval_output'
OUT.mkdir(exist_ok=True)
CODE = pathlib.Path('/tmp/image_retrieval_code')
CODE.mkdir(exist_ok=True)
'''
    for name in ('common.py', 'pipeline.py'):
        bootstrap += f"(CODE / {name!r}).write_text({(HERE / name).read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
    bootstrap += f"(CODE / 'requirements.txt').write_text({requirements!r}, encoding='utf-8')\n"
    bootstrap += '''try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv==0.9.7'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.12', '/tmp/image-retrieval-env'], check=True)
    PYTHON = '/tmp/image-retrieval-env/bin/python'
    subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON,
                    '-r', str(CODE / 'requirements.txt')], check=True)
except Exception as exc:
    (OUT / 'run_manifest.json').write_text(json.dumps({'status': 'failed', 'failed_stage': 'bootstrap',
        'images_total': 5000, 'images_completed': 0, 'images_failed': 0, 'images_pending': 5000,
        'errors': [{'type': type(exc).__name__, 'message': str(exc)}]}, indent=2))
    raise
'''
    run = '''# Optional: mount output from an earlier run and set its output directory here.
RESUME_DIR = ''
import zipfile
roots = list(pathlib.Path('/kaggle/input').rglob('manifest.jsonl'))
roots = [p.parent for p in roots if (p.parent / 'config.json').exists() and
         ((p.parent / 'images').exists() or (p.parent / 'images.zip').exists())]
if len(roots) != 1:
    raise RuntimeError(f'Expected exactly one input dataset with images/manifest; found {roots}')
DATA = roots[0]
if not (DATA / 'images').exists() and (DATA / 'images.zip').exists():
    target = pathlib.Path('/tmp/image_retrieval_input')
    target.mkdir(exist_ok=True)
    for name in ('manifest.jsonl', 'config.json'):
        import shutil
        shutil.copy2(DATA / name, target / name)
    with zipfile.ZipFile(DATA / 'images.zip') as archive:
        for member in archive.infolist():
            path = (target / member.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe archive path')
        names = archive.namelist()
        destination = target if any(name.startswith('images/') for name in names) else target / 'images'
        archive.extractall(destination)
    DATA = target
command = [PYTHON, '-u', str(CODE / 'pipeline.py'), '--root', str(DATA), '--out', str(OUT)]
if RESUME_DIR:
    command += ['--resume', RESUME_DIR]
# These checkpoints are public. Deliberately do not read Kaggle/Hugging Face secrets.
env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
env.pop('HF_TOKEN', None)
env.pop('HUGGING_FACE_HUB_TOKEN', None)
with (OUT / 'pipeline.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=env)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    code = process.wait()
if code:
    raise RuntimeError(f'Pipeline failed (exit {code}); inspect run_manifest.json and pipeline.log')
display(json.loads((OUT / 'run_manifest.json').read_text()))
'''
    review = '''from IPython.display import display, Image
for path in sorted((OUT / 'smoke_overlays').glob('*.jpg')):
    print(path.name)
    display(Image(filename=str(path)))
'''
    return bootstrap, run, review


def generate_notebook(folder, account_state):
    folder.mkdir(parents=True, exist_ok=True)
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Image retrieval preprocessing — 5,000 images\n',
        'Private dataset; public Qwen3-VL-Embedding-8B + Grounding DINO + SAM 2.1.\n',
        '20-image smoke test precedes full processing. No captioning. No secrets required.\n',
        'Resume: attach a previous output dataset and set RESUME_DIR in cell 3.\n']}]
    for source in notebook_sources():
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': source.splitlines(keepends=True)})
    for i, cell in enumerate(cells):
        cell['id'] = f'image-retrieval-{i}'
    atomic_json(folder / 'preprocess.ipynb', {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': cells})
    atomic_json(folder / 'kernel-metadata.json', {
        'id': account_state['kernel'], 'title': account_state['kernel'].split('/')[1],
        'code_file': 'preprocess.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [account_state['dataset']],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': [],
    })
    scan_package(folder)


def prepare(args):
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    rows = build_manifest(args.data, config['expected_images'])
    config['labels'] = sorted(set(config['extra_labels']) | {label for row in rows for label in json.loads(row['class'])})
    package_id = fingerprint({'config': config, 'rows': rows})
    WORK.mkdir(exist_ok=True)
    if STATE.exists():
        current = state()
        if current['package_fingerprint'] != package_id:
            raise RuntimeError('Dataset/config changed. Use a separate work directory or archive .work before creating a new resource')
    else:
        _, username = authenticate()
        suffix = uuid.uuid4().hex[:8]
        current = {'dataset': f'{username}/kitchen-images-private-{suffix}',
                   'kernel': f'{username}/kitchen-preprocess-{suffix}',
                   'package_fingerprint': package_id, 'dataset_created': False, 'kernel_pushed': False}
        atomic_json(STATE, current)
    folder = WORK / 'dataset'
    (folder / 'images').mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, 1):
        target = folder / row['path']
        if not target.exists() or digest_file(target) != row['sha256']:
            shutil.copy2(args.data / row['original_path'], target)
        if i % 1000 == 0:
            print(f'Packaged {i}/{len(rows)} images', flush=True)
    with (folder / 'manifest.jsonl').open('w', encoding='utf-8', newline='\n') as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
    atomic_json(folder / 'config.json', config)
    atomic_json(folder / 'dataset-metadata.json', {
        'id': current['dataset'], 'title': current['dataset'].split('/')[1],
        'licenses': [{'name': 'other'}],
        'description': 'Private image retrieval input: images, original metadata, validated manifest, fixed object vocabulary. Frame indices are not converted to seconds.',
    })
    scan_package(folder)
    generate_notebook(WORK / 'notebook', current)
    shutil.copy2(WORK / 'notebook' / 'preprocess.ipynb', HERE / 'preprocess.ipynb')
    atomic_json(WORK / 'validation.json', {'images': len(rows), 'unique_ids': len(set(r['imageID'] for r in rows)),
                'labels': config['labels'], 'package_fingerprint': package_id,
                'image_bytes': sum((folder / r['path']).stat().st_size for r in rows),
                'secrets_scan': 'passed'})
    print(f'Prepared {len(rows)} images and {len(config["labels"])} object labels. Package secret scan passed.')


def upload_dataset(args):
    current = state()
    api, username = authenticate()
    if not current['dataset'].startswith(username + '/'):
        raise RuntimeError('Authenticated account differs from prepared resource owner')
    if current['dataset_created']:
        print('Dataset already created by this pipeline:', current['dataset'])
        return
    scan_package(WORK / 'dataset')
    # Unique slugs prevent collisions; never create versions of unrelated datasets.
    response = api.dataset_create_new(str(WORK / 'dataset'), public=False, quiet=False,
                                      convert_to_csv=False, dir_mode='zip')
    if getattr(response, 'error', None):
        raise RuntimeError(str(response.error))
    current['dataset_created'] = True
    atomic_json(STATE, current)
    print('Private dataset created:', 'https://www.kaggle.com/datasets/' + current['dataset'])


def upload_notebook(args):
    current = state()
    if not current['dataset_created']:
        raise RuntimeError('Upload dataset first')
    if current['kernel_pushed'] and not args.rerun:
        raise RuntimeError('Notebook already pushed. Use --rerun explicitly to create another run of this pipeline notebook')
    api, username = authenticate()
    if not current['kernel'].startswith(username + '/'):
        raise RuntimeError('Authenticated account differs from resource owner')
    status = api.dataset_status(current['dataset'])
    if 'ready' not in str(status).lower():
        raise RuntimeError(f'Dataset is not ready: {status}')
    generate_notebook(WORK / 'notebook', current)
    shutil.copy2(WORK / 'notebook' / 'preprocess.ipynb', HERE / 'preprocess.ipynb')
    response = api.kernels_push(str(WORK / 'notebook'), timeout='43200', acc='NvidiaTeslaT4')
    if getattr(response, 'error', None):
        raise RuntimeError(str(response.error))
    current['kernel_pushed'] = True
    current['last_notebook_sha256'] = digest_file(WORK / 'notebook' / 'preprocess.ipynb')
    atomic_json(STATE, current)
    print('Private notebook submitted:', 'https://www.kaggle.com/code/' + current['kernel'])
    print(str(response))


def status(args):
    current = state()
    api, _ = authenticate()
    print('Dataset:', current['dataset'], api.dataset_status(current['dataset']) if current['dataset_created'] else 'not uploaded')
    if current['kernel_pushed']:
        print('Notebook:', current['kernel'], str(api.kernels_status(current['kernel'])))


def bounded_kernel_status(kernel):
    """Read-only REST status with explicit timeouts (SDK calls can wait indefinitely)."""
    import requests
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    owner, slug = kernel.split('/')
    response = requests.post(
        'https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus',
        headers={'Authorization': 'Bearer ' + os.environ['KAGGLE_API_TOKEN']},
        json={'userName': owner, 'kernelSlug': slug}, timeout=(15, 60))
    response.raise_for_status()
    result = response.json()
    if result.get('status') not in {'QUEUED', 'RUNNING', 'COMPLETE', 'ERROR', 'CANCEL_REQUESTED', 'CANCEL_ACKNOWLEDGED', 'NEW_SCRIPT'}:
        raise RuntimeError('Unrecognized Kaggle status response')
    return result


def download(args):
    current = state()
    api, _ = authenticate()
    destination = HERE / 'output'
    destination.mkdir(exist_ok=True)
    import re
    import zipfile
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    pattern = re.compile(args.pattern) if args.pattern else None
    owner, slug = current['kernel'].split('/')
    token = None
    while True:
        with api.build_kaggle_client() as client:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 100
            if token: request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            if pattern and not pattern.search(item.file_name) and item.file_name != '_output_.zip':
                continue
            target = (destination / item.file_name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe output file path')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with requests.get(item.url, stream=True, timeout=(15,120)) as download_response:
                    download_response.raise_for_status()
                    temporary = target.with_suffix(target.suffix + '.partial')
                    with temporary.open('wb') as stream:
                        for chunk in download_response.iter_content(1024*1024):
                            stream.write(chunk)
                    os.replace(temporary, target)
                print('Downloaded:', target.name, flush=True)
            if target.name == '_output_.zip':
                with zipfile.ZipFile(target) as archive:
                    for member in archive.infolist():
                        if pattern and not pattern.search(member.filename):
                            continue
                        extracted = (destination / member.filename).resolve()
                        if not extracted.is_relative_to(destination.resolve()):
                            raise ValueError('Unsafe archive member path')
                        archive.extract(member, destination)
                print('Kaggle output archive extracted', flush=True)
        if response.log:
            (destination / (slug+'.log')).write_text(response.log, encoding='utf-8')
        token = response.next_page_token
        if not token: break
    print('Outputs:', destination)


def logs(args):
    api, _ = authenticate()
    kernel = state()['kernel']
    if args.follow:
        for event in api.kernels_logs_stream(kernel):
            print(event.get('data', ''), end='', flush=True)
    else:
        print(api.kernels_logs(kernel))


def verify_private(args):
    current = state()
    api, _ = authenticate()
    folder = WORK / 'remote_metadata'
    folder.mkdir(exist_ok=True)
    api.dataset_metadata(current['dataset'], str(folder))
    dataset_meta = json.loads((folder / 'dataset-metadata.json').read_text(encoding='utf-8'))
    # New API exposes isPrivate in the fetched dataset settings.
    value = dataset_meta.get('info', dataset_meta).get('isPrivate')
    if value is not True:
        raise RuntimeError(f'Dataset privacy could not be confirmed: isPrivate={value}')
    if current['kernel_pushed']:
        notebook_dir = folder / 'notebook'
        notebook_dir.mkdir(exist_ok=True)
        api.kernels_pull(current['kernel'], str(notebook_dir), metadata=True, quiet=True)
        kernel_meta = json.loads((notebook_dir / 'kernel-metadata.json').read_text(encoding='utf-8'))
        if kernel_meta.get('is_private') not in (True, 'true'):
            raise RuntimeError('Notebook is not confirmed private')
    print('Remote privacy verified:', 'dataset + notebook' if current['kernel_pushed'] else 'dataset')


def watch(args):
    """Bounded foreground workflow; no scheduled job or model fallback."""
    deadline = time.monotonic() + args.hours * 3600
    last = None
    api = None
    failures = 0
    while time.monotonic() < deadline:
        current = state()
        progress = {'dataset': current['dataset'], 'kernel': current['kernel'],
                    'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            if not current['dataset_created']:
                message = 'Waiting for local dataset upload to finish'
            elif current['kernel_pushed']:
                response = bounded_kernel_status(current['kernel'])
                name = response['status']
                message = 'Notebook: ' + name
                progress['kernel_status'] = name
                if name in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                    progress['failure_message'] = response.get('failureMessage', '')
                    atomic_json(WORK / 'progress.json', progress)
                    print(message, progress['failure_message'], flush=True)
                    if args.download:
                        download(argparse.Namespace(pattern=None))
                        manifests = list((HERE / 'output').rglob('run_manifest.json'))
                        if len(manifests) == 1:
                            run = json.loads(manifests[0].read_text(encoding='utf-8'))
                            print('Pipeline status:', run.get('status'), flush=True)
                            if run.get('status') != 'complete':
                                raise RuntimeError('Remote pipeline did not complete; inspect downloaded run_manifest.json and pipeline.log')
                    if name != 'COMPLETE':
                        raise RuntimeError('Kaggle execution ended with ' + name)
                    return
            else:
                if api is None:
                    api, _ = authenticate()
                dataset_status = str(api.dataset_status(current['dataset']))
                if 'ready' not in dataset_status.lower():
                    message = 'Dataset processing: ' + dataset_status
                elif not current['kernel_pushed']:
                    if not args.submit_pending:
                        raise RuntimeError('Dataset ready but notebook not submitted; use upload-notebook or watch --submit-pending')
                    upload_notebook(argparse.Namespace(rerun=False))
                    verify_private(args)
                    message = 'Notebook submitted; privacy verified'
            failures = 0
            progress['message'] = message
            atomic_json(WORK / 'progress.json', progress)
            if message != last:
                print(message, flush=True)
                last = message
        except (ConnectionError, TimeoutError) as exc:
            failures += 1
            if failures >= 5:
                raise
            print('Temporary connection failure; retrying', type(exc).__name__, flush=True)
        except Exception as exc:
            # requests exceptions do not inherit builtin ConnectionError.
            import requests
            if isinstance(exc, requests.exceptions.RequestException) and failures < 4:
                failures += 1
                print('Temporary API failure; retrying', type(exc).__name__, flush=True)
            else:
                progress['error'] = type(exc).__name__
                atomic_json(WORK / 'progress.json', progress)
                raise
        time.sleep(30)
    raise TimeoutError('Watch deadline reached; remote job is not cancelled. Use status/download to inspect it')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.set_defaults(func=prepare)
    for name, action in [('upload-dataset', upload_dataset), ('upload-notebook', upload_notebook),
                         ('status', status), ('download', download), ('verify-private', verify_private),
                         ('watch', watch), ('logs', logs)]:
        p = sub.add_parser(name)
        if name == 'upload-notebook':
            p.add_argument('--rerun', action='store_true')
        if name == 'download':
            p.add_argument('--pattern', default=None)
        if name == 'watch':
            p.add_argument('--submit-pending', action='store_true')
            p.add_argument('--download', action='store_true')
            p.add_argument('--hours', type=float, default=16)
        if name == 'logs':
            p.add_argument('--follow', action='store_true')
        p.set_defaults(func=action)
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        # Do not emit auth headers, environment values or credential tracebacks.
        message = str(exc)
        token = os.environ.get('KAGGLE_API_TOKEN', '')
        if token:
            message = message.replace(token, '[REDACTED]')
        print(f'{type(exc).__name__}: {message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
