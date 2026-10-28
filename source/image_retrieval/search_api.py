"""Image-first retrieval API over verified Kaggle artifacts. No VBS adapters."""
from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache
import io
import json
import os
from pathlib import Path
from typing import Literal
import secrets
import threading

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from common import digest_file, validate_vectors

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


class ObjectFilter(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    color: str | None = None
    min_count: int = Field(default=1, ge=0, le=100)
    max_count: int | None = Field(default=None, ge=0, le=100)
    count_source: Literal['detected', 'metadata'] = 'detected'
    min_color_fraction: float = Field(default=.15, ge=0, le=1)

    @model_validator(mode='after')
    def validate_count(self):
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError('Số lượng tối đa phải ≥ số lượng tối thiểu')
        if self.count_source == 'metadata' and self.color:
            raise ValueError('Metadata gốc không có màu theo instance; dùng detection cho điều kiện có màu')
        return self


class SearchRequest(BaseModel):
    text: str = Field(default='', max_length=2000)
    reference_id: str | None = None
    query_asset: str | None = None
    day: str | None = None
    source: str | None = None
    hour_from: int | None = Field(default=None, ge=0, le=23)
    hour_to: int | None = Field(default=None, ge=0, le=23)
    objects: list[ObjectFilter] = Field(default_factory=list, max_length=12)
    shortlist_ids: list[str] | None = Field(default=None, max_length=5000)
    positive_ids: list[str] = Field(default_factory=list, max_length=100)
    negative_ids: list[str] = Field(default_factory=list, max_length=100)
    rerank: bool = False
    rerank_k: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=60, ge=1, le=120)

    @model_validator(mode='after')
    def validate_range(self):
        if self.hour_from is not None and self.hour_to is not None and self.hour_from > self.hour_to:
            raise ValueError('Giờ bắt đầu phải nhỏ hơn hoặc bằng giờ kết thúc')
        if self.reference_id and self.query_asset:
            raise ValueError('Chọn một ảnh tham chiếu hoặc một ảnh upload')
        if set(self.positive_ids) & set(self.negative_ids):
            raise ValueError('Ảnh không thể vừa like vừa dislike')
        return self


def feedback_vector(cat, query, vector):
    """Rocchio feedback over stored vectors; recompute from the original query each time."""
    result = np.zeros(cat.vectors.shape[1], dtype=np.float32)
    if vector is not None:
        norm = np.linalg.norm(vector)
        if norm > 1e-8: result += vector / norm
    for ids, weight in ((query.positive_ids, .75), (query.negative_ids, -.35)):
        unique = list(dict.fromkeys(ids))
        if any(i not in cat.index for i in unique): raise HTTPException(404,'Ảnh feedback không tồn tại')
        if unique: result += weight * np.mean(cat.vectors[[cat.index[i] for i in unique]],axis=0)
    norm = np.linalg.norm(result)
    if norm < 1e-8: raise HTTPException(422,'Feedback triệt tiêu query. Hãy bỏ một lựa chọn.')
    return result / norm


class Catalog:
    def __init__(self, artifacts: Path, data: Path):
        self.root = artifacts.resolve()
        self.data = data.resolve()
        manifest = json.loads((self.root / 'run_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'complete' or manifest.get('images_failed') != 0:
            raise ValueError('Preprocessing artifacts are not complete')
        for name, expected in manifest['outputs'].items():
            if digest_file(self.root / name) != expected:
                raise ValueError(f'Artifact checksum mismatch: {name}')
        self.rows = pq.read_table(self.root / 'images.parquet').to_pylist()
        self.ids = json.loads((self.root / 'embedding_ids.json').read_text(encoding='utf-8'))
        if self.ids != [r['imageID'] for r in self.rows] or len(set(self.ids)) != len(self.ids):
            raise ValueError('Embedding IDs do not match metadata row order')
        self.vectors = np.load(self.root / 'image_embeddings.npy', mmap_mode='r')
        validate_vectors(self.vectors, len(self.ids), 4096)
        self.index = {key: i for i, key in enumerate(self.ids)}
        self.metadata_counts = {r['imageID']: json.loads(r.get('class_count', '{}')) for r in self.rows}
        self.config = json.loads((self.root / 'config.json').read_text(encoding='utf-8'))
        self.runtime = json.loads((self.root / 'embedding_runtime.json').read_text(encoding='utf-8'))
        if self.runtime['revision'] != self.config['embedding']['revision']:
            raise ValueError('Embedding runtime/model revision mismatch')
        self.objects = defaultdict(list)
        for obj in pq.read_table(self.root / 'objects.parquet').to_pylist():
            if obj['imageID'] not in self.index:
                raise ValueError('Unknown image in object table')
            obj['colors'] = json.loads(obj.pop('colors_json'))
            self.objects[obj['imageID']].append(obj)
        self.labels = sorted({o['label'] for rows in self.objects.values() for o in rows})
        self.days = sorted({r['day'] for r in self.rows})
        self.sources = sorted({r['source'] for r in self.rows})
        self.groups = defaultdict(list)
        for row in self.rows:
            self.groups[(row['day'], row['source'], row['hour'])].append(row['imageID'])
        self.positions = {}
        for key, ids in self.groups.items():
            ids.sort(key=lambda i: self.rows[self.index[i]]['frame'])
            for position, image_id in enumerate(ids):
                self.positions[image_id] = (key, position)
        # Fail early if the local/original image mapping is not usable.
        for row in self.rows:
            self.image_path(row['imageID'])

    def image_path(self, image_id):
        if image_id not in self.index:
            raise KeyError(image_id)
        row = self.rows[self.index[image_id]]
        for relative in (row['original_path'], row['path']):
            candidate = (self.data / relative).resolve()
            if not candidate.is_relative_to(self.data):
                raise ValueError('Image path escapes data root')
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f'Missing image: {image_id}')

    def hit(self, image_id, score=None):
        row = self.rows[self.index[image_id]]
        objects = self.objects[image_id]
        counts = defaultdict(int)
        for obj in objects:
            counts[obj['label']] += 1
        return {'imageID': image_id, 'day': row['day'], 'hour': row['hour'], 'frame': row['frame'],
                'videoID': f"{row['day']}_{row['source']}_{row['hour']:02d}",
                'source': row['source'], 'batch': row['batch'], 'score': score,
                'thumbnailUrl': f'/api/images/media/{image_id}?thumbnail=true',
                'imageUrl': f'/api/images/media/{image_id}',
                'objectCounts': dict(counts), 'metadataObjectCounts': getattr(self, 'metadata_counts', {}).get(image_id, {})}

    def object_count(self, image_id, f):
        if f.count_source == 'metadata':
            # Missing metadata labels are unknown, never proof that the count is zero.
            counts = getattr(self, 'metadata_counts', {}).get(image_id, {})
            if f.label not in counts:
                return None
            return counts[f.label]
        return sum(1 for obj in self.objects[image_id] if obj['label'] == f.label and
                   (not f.color or (obj['colors'].get(f.color, 0) > 0 and obj['colors'].get(f.color, 0) >= f.min_color_fraction)))

    def count_matches(self, image_id, f):
        count = self.object_count(image_id, f)
        if count is None: return False
        return count >= f.min_count and (f.max_count is None or count <= f.max_count)

    def search(self, query: SearchRequest, vector=None):
        candidates = []
        allowed = set(query.shortlist_ids) if query.shortlist_ids is not None else None
        for i, row in enumerate(self.rows):
            image_id = row['imageID']
            if allowed is not None and image_id not in allowed:
                continue
            if query.day and row['day'] != query.day or query.source and row['source'] != query.source:
                continue
            if query.hour_from is not None and row['hour'] < query.hour_from or query.hour_to is not None and row['hour'] > query.hour_to:
                continue
            if any(not self.count_matches(image_id, f) for f in query.objects):
                continue
            candidates.append(i)
        scores = {}
        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (4096,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
                raise ValueError('Invalid query embedding')
            vector = vector / np.linalg.norm(vector)
            if candidates:
                values = np.asarray(self.vectors[candidates]) @ vector
                scores = {i: float(v) for i, v in zip(candidates, values)}
                candidates.sort(key=lambda i: (-scores[i], i))
        selected = candidates[query.offset:query.offset+query.limit]
        return {'total': len(candidates), 'offset': query.offset, 'limit': query.limit,
                'ranked': vector is not None,
                'items': [{**self.hit(self.ids[i], scores.get(i)),
                           'constraintCounts':[{'label':f.label,'color':f.color,'source':f.count_source,
                                                'count':self.object_count(self.ids[i],f)} for f in query.objects]}
                          for i in selected]}


class QueryEncoder:
    def __init__(self, catalog, gpu=None):
        self.catalog = catalog
        self.gpu = gpu
        self.mode = os.environ.get('IMAGE_ENCODER_MODE', 'remote' if os.environ.get('IMAGE_ENCODER_URL') else 'disabled')
        self.url = os.environ.get('IMAGE_ENCODER_URL', '').rstrip('/')
        self.key = os.environ.get('IMAGE_ENCODER_KEY', '')
        self.model = None
        self.lock = threading.Lock()

    @lru_cache(maxsize=128)
    def encode(self, text='', image=None):
        with self.lock:
            if self.mode == 'local':
                item = {}
                if text:
                    item['text'] = text
                if image:
                    item['image'] = str(image)
                return self.gpu.encode([item])[0]
            if self.mode == 'remote' and self.url:
                import requests
                with io.BytesIO(image.read_bytes()) if image else io.BytesIO() as data:
                    response = requests.post(self.url + '/api/images/encode',
                        data={'text': text}, files={'image': (image.name, data, 'application/octet-stream')} if image else None,
                        headers={'Authorization': 'Bearer ' + self.key} if self.key else {}, timeout=(15, 180))
                response.raise_for_status()
                payload = response.json()
                if payload.get('embedding_config') != self.catalog.config['embedding']:
                    raise ValueError('Remote encoder configuration differs from dataset embeddings')
                vector = np.asarray(payload['vector'], dtype=np.float32)
                validate_vectors(vector[None, :], 1, 4096)
                return vector
            raise RuntimeError('Chưa kết nối Qwen query encoder. Có thể lọc dữ liệu hoặc chọn ảnh trong tập làm query.')


def create_app(artifacts=None, data=None):
    if artifacts is None:
        artifacts = Path(os.environ.get('IMAGE_ARTIFACT_DIR', HERE / 'output' / 'image_retrieval_output'))
    if data is None:
        data = Path(os.environ.get('IMAGE_DATA_DIR', PROJECT / 'data'))
    @asynccontextmanager
    async def lifespan(app):
        app.state.catalog = Catalog(Path(artifacts), Path(data))
        from reranker import GPUModels, QueryReranker
        app.state.gpu = GPUModels(app.state.catalog.config['embedding'])
        app.state.encoder = QueryEncoder(app.state.catalog,app.state.gpu)
        app.state.reranker = QueryReranker(app.state.catalog,app.state.gpu)
        app.state.uploads = {}
        yield

    app = FastAPI(title='Image Retrieval', lifespan=lifespan)
    key = os.environ.get('IMAGE_API_KEY', '')

    @app.middleware('http')
    async def protect(request: Request, call_next):
        if key and request.url.path.startswith('/api/images'):
            if not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + key):
                return Response('Unauthorized', status_code=401)
        return await call_next(request)

    def catalog():
        return app.state.catalog

    from dres import install_dres
    install_dres(app, catalog)

    @app.get('/api/images/health')
    def health():
        cat = catalog()
        return {'status': 'ok', 'images': len(cat.ids), 'objects': sum(map(len, cat.objects.values())),
                'queryEncoder': app.state.encoder.mode, 'queryReranker':app.state.reranker.mode,
                'model': cat.config['embedding']['model_id']}

    @app.get('/api/images/facets')
    def facets():
        cat = catalog()
        return {'days': cat.days, 'sources': cat.sources, 'labels': cat.labels,
                'colors': cat.config['colors'], 'hours': sorted({r['hour'] for r in cat.rows})}

    @app.post('/api/images/agent/parse')
    def parse_query(payload: dict):
        from query_agent import AgentInput, QueryAgent, AgentUnavailable
        try:
            request = AgentInput.model_validate(payload)
        except ValueError as exc:
            raise HTTPException(422, 'Yêu cầu parser không hợp lệ') from exc
        cat = catalog()
        context = {'labels': cat.labels, 'colors': cat.config['colors'], 'days': cat.days, 'sources': cat.sources}
        try:
            return QueryAgent.instance().parse(request, context)
        except AgentUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post('/api/images/query-assets')
    async def upload(image: UploadFile = File(...)):
        content = await image.read(12 * 1024 * 1024 + 1)
        if len(content) > 12 * 1024 * 1024:
            raise HTTPException(413, 'Ảnh query tối đa 12 MB')
        try:
            with Image.open(io.BytesIO(content)) as opened:
                if opened.width * opened.height > 40_000_000:
                    raise ValueError('Ảnh quá lớn')
                opened.load()
                converted = ImageOps.exif_transpose(opened).convert('RGB')
        except Exception:
            raise HTTPException(400, 'Ảnh query không hợp lệ')
        directory = HERE / '.work' / 'query-assets'
        directory.mkdir(parents=True, exist_ok=True)
        asset = secrets.token_hex(16)
        path = directory / (asset + '.jpg')
        converted.save(path, quality=95)
        app.state.uploads[asset] = path
        return {'asset': asset}

    @app.post('/api/images/search')
    def search(query: SearchRequest):
        cat = catalog()
        vector = None
        if query.reference_id:
            if query.reference_id not in cat.index:
                raise HTTPException(404, 'Không tìm thấy ảnh tham chiếu')
            vector = np.asarray(cat.vectors[cat.index[query.reference_id]])
        image = None
        if query.query_asset:
            image = app.state.uploads.get(query.query_asset)
            if image is None:
                raise HTTPException(404, 'Ảnh query hết phiên; hãy upload lại')
        if query.text.strip() or image:
            try:
                encoded = app.state.encoder.encode(query.text.strip(), image)
                vector = encoded if vector is None else vector + encoded
            except Exception as exc:
                # Do not leak remote URLs/credentials in HTTP error responses.
                raise HTTPException(503, 'Qwen query encoder chưa sẵn sàng. Kiểm tra kết nối encoder và log backend.') from exc
        if query.positive_ids or query.negative_ids:
            vector = feedback_vector(cat,query,vector)
        if query.rerank:
            if app.state.reranker.mode=='disabled': raise HTTPException(503,'Qwen reranker chưa được bật.')
            rerank_image=image or (cat.image_path(query.reference_id) if query.reference_id else None)
            if not query.text.strip() and rerank_image is None:
                raise HTTPException(422,'Reranker cần mô tả hoặc ảnh query.')
            try:
                from reranker import rerank_page
                return rerank_page(cat,query,vector,app.state.reranker,rerank_image)
            except Exception as exc:
                raise HTTPException(503,'Qwen reranker chưa sẵn sàng. Kiểm tra cấu hình GPU/reranker.') from exc
        return cat.search(query, vector)

    @app.post('/api/images/encode')
    async def encode(request: Request):
        if app.state.encoder.mode != 'local':
            raise HTTPException(503, 'Instance này không chạy Qwen encoder')
        form = await request.form()
        text = str(form.get('text', ''))
        image = form.get('image')
        path = None
        if image:
            asset = await upload(image)
            path = app.state.uploads[asset['asset']]
        import asyncio
        vector = await asyncio.to_thread(app.state.encoder.encode, text, path)
        return {'vector': vector.tolist(), 'embedding_config': catalog().config['embedding']}

    @app.get('/api/images/media/{image_id}')
    def media(image_id: str, thumbnail: bool = False):
        try:
            path = catalog().image_path(image_id)
        except KeyError:
            raise HTTPException(404, 'Ảnh không tồn tại')
        if not thumbnail:
            return FileResponse(path)
        return Response(thumbnail_bytes(str(path)), media_type='image/jpeg', headers={'Cache-Control':'private,max-age=86400'})

    @app.get('/api/images/detail/{image_id}')
    def detail(image_id: str):
        cat = catalog()
        if image_id not in cat.index:
            raise HTTPException(404, 'Ảnh không tồn tại')
        key, pos = cat.positions[image_id]
        nearby = cat.groups[key][max(0, pos-4):pos+5]
        return {**cat.hit(image_id), 'objects': cat.objects[image_id],
                'neighbors': [cat.hit(i) for i in nearby],
                'videoFrameCount': len(cat.groups[key]), 'videoFramePosition': pos,
                'previousImageID': cat.groups[key][pos-1] if pos else None,
                'nextImageID': cat.groups[key][pos+1] if pos+1 < len(cat.groups[key]) else None,
                'originalDescription': cat.rows[cat.index[image_id]].get('description', '')}

    @app.get('/api/images/video/{image_id}')
    def video_frames(image_id: str):
        cat = catalog()
        if image_id not in cat.positions: raise HTTPException(404, 'Ảnh không tồn tại')
        key, _ = cat.positions[image_id]
        return {'videoID':cat.hit(image_id)['videoID'],'items':[cat.hit(i) for i in cat.groups[key]]}

    @app.get('/api/images/mask/{instance_id}')
    def mask(instance_id: str):
        # Only exact instances present in the catalog can be served.
        image_id = instance_id.rsplit('__', 1)[0]
        obj = next((o for o in catalog().objects.get(image_id, []) if o['instance_id'] == instance_id), None)
        if obj is None:
            raise HTTPException(404)
        path = (catalog().root / obj['mask_path']).resolve()
        if not path.is_relative_to(catalog().root):
            raise HTTPException(400)
        return FileResponse(path, media_type='image/png')

    dist = PROJECT / 'frontend' / 'dist'
    if (dist / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')
        @app.get('/')
        @app.get('/images')
        def index():
            return FileResponse(dist / 'index.html')
    return app


@lru_cache(maxsize=256)
def thumbnail_bytes(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        image.thumbnail((480, 320))
        out = io.BytesIO()
        image.save(out, format='JPEG', quality=82)
        return out.getvalue()


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(PROJECT / '.env', override=False)
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get('IMAGE_API_HOST', '127.0.0.1'),
                port=int(os.environ.get('IMAGE_API_PORT', '8100')))
