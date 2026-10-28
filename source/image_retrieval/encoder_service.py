"""Ephemeral authenticated Qwen query encoder for a private Kaggle notebook."""
import asyncio
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps
import requests
import torch
import uvicorn

from reranker import GPUModels, MODEL, REVISION


def main():
    directory = Path(__file__).resolve().parent
    config = json.loads((directory / 'config.json').read_text())['embedding']
    if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('Query encoder requires exactly 2 T4 GPUs')
    print('Loading Qwen query encoder on two T4 GPUs', flush=True)
    encoder = GPUModels(config)
    encoder.encode([{'text': 'A person in a kitchen.'}])
    key = secrets.token_urlsafe(32)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()
    cache = OrderedDict()

    @app.middleware('http')
    async def authenticate(request: Request, call_next):
        if not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + key):
            return Response('Unauthorized', status_code=401)
        return await call_next(request)

    @app.get('/health')
    def health():
        return {'ready': True, 'embedding_config': config, 'reranker_model':MODEL,'reranker_revision':REVISION}

    @app.post('/api/images/rerank')
    async def rerank(text: str = Form(''), images: list[UploadFile] = File(...), query_image: UploadFile | None = File(None)):
        if len(text)>2000 or not 1<=len(images)<=50:
            raise HTTPException(422,'Rerank accepts up to 50 images and 2000 text characters')
        total=0
        async def read_image(upload):
            nonlocal total
            content=await upload.read(4*1024*1024+1)
            total+=len(content)
            if len(content)>4*1024*1024 or total>64*1024*1024: raise HTTPException(413,'Rerank upload too large')
            try:
                with Image.open(io.BytesIO(content)) as opened:
                    if opened.width*opened.height>40_000_000: raise ValueError('Image too large')
                    opened.load()
                    return ImageOps.exif_transpose(opened).convert('RGB')
            except Exception: raise HTTPException(400,'Invalid image')
        rgb=await read_image(query_image) if query_image else None
        documents=[await read_image(upload) for upload in images]
        if not text.strip() and rgb is None: raise HTTPException(422,'Provide query text or image')
        scores=await asyncio.to_thread(encoder.score,text.strip(),rgb,documents)
        return {'model':MODEL,'revision':REVISION,'scores':scores}

    @app.post('/api/images/encode')
    async def encode(text: str = Form(''), image: UploadFile | None = File(None)):
        if len(text) > 2000:
            raise HTTPException(400, 'Query text is too long')
        content = None
        rgb = None
        if image:
            content = await image.read(12 * 1024 * 1024 + 1)
            if len(content) > 12 * 1024 * 1024:
                raise HTTPException(413, 'Query image exceeds 12 MB')
            try:
                with Image.open(io.BytesIO(content)) as opened:
                    if opened.width * opened.height > 40_000_000:
                        raise ValueError('Image too large')
                    opened.load()
                    rgb = ImageOps.exif_transpose(opened).convert('RGB')
            except Exception:
                raise HTTPException(400, 'Invalid image')
        if not text.strip() and rgb is None:
            raise HTTPException(400, 'Provide text or an image')
        cache_key = hashlib.sha256(text.encode() + b'\0' + (content or b'')).hexdigest()
        def calculate():
            with lock:
                if cache_key in cache:
                    cache.move_to_end(cache_key)
                    return cache[cache_key]
                item = {}
                if text.strip(): item['text'] = text.strip()
                if rgb is not None: item['image'] = rgb
                vector = encoder.encode([item])[0].tolist()
                cache[cache_key] = vector
                if len(cache) > 128: cache.popitem(last=False)
                return vector
        vector = await asyncio.to_thread(calculate)
        return {'vector': vector, 'embedding_config': config}

    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=8111, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(60):
        if server.started: break
        time.sleep(1)
    else: raise RuntimeError('Query API failed to start')

    release = json.loads((directory / 'cloudflared_release.json').read_text())
    binary = Path('/tmp/cloudflared-image-encoder')
    response = requests.get(release['url'], timeout=(15,120))
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != release['sha256']:
        raise RuntimeError('cloudflared checksum mismatch')
    binary.write_bytes(response.content)
    binary.chmod(0o700)
    tunnel = subprocess.Popen([str(binary), 'tunnel', '--url', 'http://127.0.0.1:8111', '--no-autoupdate'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = None
        for line in tunnel.stdout:
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
            if match:
                url = match.group(0)
                break
        if not url: raise RuntimeError('Quick Tunnel did not return a URL')
        connection = {'url': url, 'key': key, 'expires_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time()+7200))}
        Path('/kaggle/working/encoder_connection.json').write_text(json.dumps(connection))
        # This marker is only in the private notebook log. The local connector
        # stores its key in ignored .env; never forward it to the browser.
        print('IMAGE_ENCODER_CONNECTION:' + json.dumps(connection), flush=True)
        deadline = time.monotonic()+7200
        while time.monotonic() < deadline and tunnel.poll() is None and thread.is_alive():
            time.sleep(10)
    finally:
        tunnel.terminate()
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == '__main__': main()
