"""Download the public image/embedding/object bundle and verify it before installing."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import zipfile
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def extract_bundle(archive, destination, expected_tree=None, expected_files=None):
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or not relative.parts or '..' in relative.parts or '\\' in entry.filename or ':' in entry.filename:
                raise ValueError('Unexpected bundle path')
            path = root.joinpath(*relative.parts)
            if not path.is_relative_to(root) or relative.parts[0] not in ('data', 'artifacts'):
                raise ValueError('Unexpected bundle path')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlinks are not allowed')
        entries = sorted((e for e in z.infolist() if not e.is_dir()), key=lambda e: e.filename)
        if len({e.filename for e in entries}) != len(entries): raise ValueError('Duplicate bundle files')
        if expected_files is not None and len(entries) != expected_files: raise ValueError('Bundle file count mismatch')
        tree = hashlib.sha256()
        parents = set()
        print(f'Extracting {len(entries)} files...', flush=True)
        for i, entry in enumerate(entries, 1):
            target = root / entry.filename
            if target.parent not in parents:
                target.parent.mkdir(parents=True, exist_ok=True)
                parents.add(target.parent)
            h = hashlib.sha256()
            with z.open(entry) as source, target.open('wb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    h.update(block)
            tree.update((entry.filename + '\0' + h.hexdigest() + '\n').encode())
            if i % 20000 == 0: print(f'Extracted {i} files', flush=True)
        if expected_tree and tree.hexdigest() != expected_tree: raise ValueError('Bundle content checksum mismatch')
        return tree.hexdigest()


def main():
    load_dotenv(ROOT / '.env', override=False)
    if not (ROOT / '.env').exists(): shutil.copyfile(ROOT / '.env.example', ROOT / '.env')
    release = json.loads((HERE / 'public_data.json').read_text())
    data = Path(os.environ.get('IMAGE_DATA_DIR') or ROOT / 'data').resolve()
    artifacts = Path(os.environ.get('IMAGE_ARTIFACT_DIR') or HERE / 'output' / 'image_retrieval_output').resolve()
    marker = artifacts / '.public_bundle_sha256'
    if marker.is_file() and marker.read_text().strip() == release['sha256'] and data.is_dir():
        print('Public dataset is already installed.')
        return
    for path in (data, artifacts):
        if path.exists() and (not path.is_dir() or any(path.iterdir())): raise RuntimeError(f'Destination is not empty: {path}')
    work = HERE / '.work' / 'public_download'
    work.mkdir(parents=True, exist_ok=True)
    bundle = work / 'bundle.zip'
    if not bundle.exists() or not zipfile.is_zipfile(bundle):
        url = release.get('download_url') or 'https://www.kaggle.com/api/v1/datasets/download/' + release['dataset']
        kaggle_download = urlsplit(url).hostname in ('www.kaggle.com', 'api.kaggle.com')
        headers = {}
        token = os.environ.get('KAGGLE_API_TOKEN', '').strip()
        if token and kaggle_download: headers['Authorization'] = 'Bearer ' + token
        package = work / 'download.zip'
        partial = work / 'download.part'
        print('Downloading public dataset:', release['dataset'], flush=True)
        for attempt in range(6):
            response = requests.get(url, headers=headers, params={'datasetVersionNumber':release.get('dataset_version',1)} if kaggle_download else None, stream=True, timeout=(20, 180))
            if response.status_code not in (404, 503) or attempt == 5: break
            response.close()
            print('Waiting for public download archive...', flush=True)
            time.sleep(10)
        with response:
            if response.status_code in (401, 403):
                raise RuntimeError('Fill KAGGLE_API_TOKEN in .env, then retry.' if kaggle_download else 'Public download is unavailable; retry later or use the Kaggle mirror in README.')
            response.raise_for_status()
            downloaded = 0
            next_report = 128 * 1024 * 1024
            with partial.open('wb') as target:
                for block in response.iter_content(1024 * 1024):
                    target.write(block)
                    downloaded += len(block)
                    if downloaded >= next_report:
                        print(f'Downloaded {downloaded // (1024 * 1024)} MiB', flush=True)
                        next_report += 128 * 1024 * 1024
        partial.replace(package)
        # Kaggle may keep the upload zip or repack its extracted data/artifacts.
        if digest(package) == release['sha256']:
            shutil.copyfile(package, bundle)
        else:
            with zipfile.ZipFile(package) as z:
                candidates = [n for n in z.namelist() if n == 'bundle.zip']
                if len(candidates) == 1:
                    with z.open(candidates[0]) as src, bundle.open('wb') as target: shutil.copyfileobj(src, target)
                elif release.get('tree_sha256'):
                    shutil.copyfile(package, bundle)
                else: raise ValueError('Public dataset does not contain bundle.zip')
    if not release.get('tree_sha256') and digest(bundle) != release['sha256']: raise ValueError('Bundle checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='install-', dir=work) as tmp:
        staging = Path(tmp)
        extract_bundle(bundle, staging, release.get('tree_sha256'), release.get('files'))
        run = json.loads((staging / 'artifacts' / 'run_manifest.json').read_text())
        for name, expected in run['outputs'].items():
            if digest(staging / 'artifacts' / name) != expected: raise ValueError('Artifact checksum mismatch')
        for source, target in ((staging / 'data', data), (staging / 'artifacts', artifacts)):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): target.rmdir()  # Only empty directories, checked above.
            shutil.move(str(source), str(target))
    marker.write_text(release['sha256'] + '\n')
    print(f"Ready: {release['images']} images, {release['objects']} objects.")


if __name__ == '__main__': main()
