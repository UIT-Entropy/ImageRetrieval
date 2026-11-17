"""Route retrieval across the configured Kaggle workers and named vectors."""
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
import threading
import time

import numpy as np
import requests
from fastapi import HTTPException

from build_index import COLLECTION
from common import validate_vectors
from model_registry import EMBED_2B, EMBED_8B, MODELS, RERANK_2B, RERANK_8B, SMALL_MODELS, fingerprint
from reranker import validate_scores


class MultiSearch:
    def __init__(self, catalog, client, work):
        self.catalog = catalog
        self.client = client
        self.work = work
        count = int(os.environ.get('IMAGE_GPU_WORKERS', '2'))
        if count not in (2, 3):
            raise ValueError('IMAGE_GPU_WORKERS must be 2 or 3')
        self.routes_by_model = self.routes(count)
        self.executor = ThreadPoolExecutor(max_workers=10)
        # Each executor thread gets its own keep-alive connection pool. requests.Session
        # is not shared across threads, and the Kaggle tunnel benefits from reusing TLS.
        self._session_local = threading.local()
        self._sessions = []
        self._sessions_lock = threading.Lock()
        self._embedding_cache = OrderedDict()
        self._embedding_cache_lock = threading.Lock()

    def _session(self):
        session = getattr(self._session_local, 'session', None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._session_local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self._sessions_lock:
            sessions, self._sessions = self._sessions, []
        for session in sessions:
            session.close()

    def _encode(self, name, text, image):
        image_digest = hashlib.sha256(image.read_bytes()).hexdigest() if image else ''
        cache_key = (name, text, image_digest)
        with self._embedding_cache_lock:
            vector = self._embedding_cache.get(cache_key)
            if vector is not None:
                self._embedding_cache.move_to_end(cache_key)
                return vector

        result = self.worker(name, 'encode', text, image)
        vector = np.asarray(result['vector'], dtype=np.float32)
        validate_vectors(vector[None, :], 1, MODELS[name]['dimension'])
        vector.setflags(write=False)
        with self._embedding_cache_lock:
            self._embedding_cache[cache_key] = vector
            self._embedding_cache.move_to_end(cache_key)
            if len(self._embedding_cache) > 128:
                self._embedding_cache.popitem(last=False)
        return vector

    @staticmethod
    def routes(count):
        routes = {EMBED_8B: (1, 'qwen8b'), RERANK_8B: (1, 'qwen8b')}
        for name in (EMBED_2B, RERANK_2B, *SMALL_MODELS):
            if count == 2:
                routes[name] = (2, 'combined')
            elif name in SMALL_MODELS:
                routes[name] = (3, 'small')
            else:
                routes[name] = (2, 'qwen2b')
        return routes

    def worker(self, name, operation, text, image, documents=()):
        slot, group = self.routes_by_model[name]
        descriptor = self.work / f'worker_{slot}_{group}.json'
        if not descriptor.is_file():
            descriptor = self.work / f'worker_{slot}.json'
        connection = json.loads(descriptor.read_text(encoding='utf-8'))
        if connection['group'] != group:
            raise ValueError('Worker topology mismatch')
        with ExitStack() as stack:
            session = self._session()
            files = [('images', (p.name, stack.enter_context(p.open('rb')), 'image/jpeg')) for p in documents]
            if image:
                field = 'query_image' if documents else 'image'
                files.append((field, (image.name, stack.enter_context(image.open('rb')), 'image/jpeg')))
            started = time.perf_counter()
            response = session.post(connection['url'] + '/api/images/' + operation,
                data={'model': name, 'text': text}, files=files,
                headers={'Authorization': 'Bearer ' + connection['key']}, timeout=(8, 90 if documents else 20))
            response.raise_for_status()
            result = response.json()
            if os.environ.get('IMAGE_DEBUG_TIMING') == '1':
                print(f"TIMING worker_rpc model={name} op={operation} seconds={time.perf_counter()-started:.3f}", flush=True)
        if result.get('model') != name or result.get('fingerprint') != fingerprint(name):
            raise ValueError('Worker and catalog model fingerprints differ')
        return result

    def rank(self, name, query, image, candidates, stored):
        from qdrant_client import models

        result_limit = query.offset + query.limit
        rerank = query.rerank_model or ('2b' if query.rerank else 'off')
        if rerank != 'off' and query.offset < query.rerank_k:
            result_limit = max(result_limit, query.rerank_k)

        vectors = []
        if query.reference_id:
            vectors.append(np.asarray(stored[self.catalog.index[query.reference_id]][name]))
        if query.text.strip() or image:
            encoded = self._encode(name, query.text.strip(), image)
            vectors.append(encoded)
        value = np.zeros(MODELS[name]['dimension'], dtype=np.float32)
        if vectors:
            base = np.sum(vectors, axis=0)
            norm = np.linalg.norm(base)
            if norm > 1e-8:
                value += base / norm
        for ids, weight in ((query.positive_ids, .75), (query.negative_ids, -.35)):
            if ids:
                value += weight * np.mean([stored[self.catalog.index[i]][name] for i in dict.fromkeys(ids)], axis=0)
        norm = np.linalg.norm(value)
        if norm < 1e-8:
            raise HTTPException(422, 'Feedback triệt tiêu query. Hãy bỏ một lựa chọn.')
        query_filter = None
        if len(candidates) != len(self.catalog.ids):
            query_filter = models.Filter(must=[models.HasIdCondition(has_id=candidates)])
        started = time.perf_counter()
        points = self.client.query_points(COLLECTION, query=(value/norm).tolist(), using=name,
            query_filter=query_filter, limit=min(len(candidates), result_limit)).points
        ranking = sorted(((int(p.id), p.score) for p in points), key=lambda pair: (-pair[1], pair[0]))
        if os.environ.get('IMAGE_DEBUG_TIMING') == '1':
            print(f"TIMING qdrant model={name} candidates={len(candidates)} limit={min(len(candidates), result_limit)} seconds={time.perf_counter()-started:.3f}", flush=True)
        return ranking

    def search(self, query, image):
        catalog = self.catalog
        ids = [*query.positive_ids, *query.negative_ids]
        if query.reference_id:
            ids.append(query.reference_id)
        if any(image_id not in catalog.index for image_id in ids):
            raise HTTPException(404, 'Không tìm thấy ảnh tham chiếu hoặc feedback')
        ranked = bool(ids or image or query.text.strip())
        if not ranked:
            if query.rerank or query.rerank_model in ('2b', '8b'):
                raise HTTPException(422, 'Reranker cần mô tả hoặc ảnh query.')
            return catalog.search(query)
        candidates = catalog.candidates(query)
        if not candidates:
            return catalog.results(query, [], {}, ranked=True)
        names = {'fast': ('siglip2',), '2b': (EMBED_2B,), '8b': (EMBED_8B,),
                 'ensemble': tuple(query.embedding_models or SMALL_MODELS)}[query.search_mode or 'fast']
        stored = {}
        if ids:
            stored = {int(p.id): p.vector for p in self.client.retrieve(
                COLLECTION, ids=list({catalog.index[i] for i in ids}), with_vectors=list(names))}
        failed = []
        rankings = []
        futures = {name: self.executor.submit(self.rank, name, query, image, candidates, stored) for name in names}
        for name, future in futures.items():
            try:
                rankings.append(future.result())
            except (OSError, ValueError, requests.RequestException):
                failed.append(name)
        if not rankings:
            raise HTTPException(503, 'Model chưa sẵn sàng: ' + ', '.join(failed))
        if len(rankings) == 1:
            scores = dict(rankings[0])
        else:
            scores = defaultdict(float)
            for ranking in rankings:
                for position, (i, _) in enumerate(ranking, 1):
                    scores[i] += 1 / (60 + position)
        order = sorted(scores, key=lambda i: (-scores[i], i))
        rerank = query.rerank_model or ('2b' if query.rerank else 'off')
        rerank_scores = {}
        rerank_name = None
        if rerank != 'off' and query.offset < query.rerank_k:
            rerank_name = RERANK_8B if rerank == '8b' else RERANK_2B
            query_image = image or (catalog.image_path(query.reference_id) if query.reference_id else None)
            if not query.text.strip() and not query_image:
                raise HTTPException(422, 'Reranker cần mô tả hoặc ảnh query.')
            head = order[:query.rerank_k]
            try:
                result = self.worker(rerank_name, 'rerank', query.text.strip(), query_image,
                                     [catalog.image_path(catalog.ids[i]) for i in head])
                rerank_scores = dict(zip(head, validate_scores(result['scores'], len(head))))
            except (OSError, ValueError, requests.RequestException) as exc:
                raise HTTPException(503, f'Model chưa sẵn sàng: {rerank_name}') from exc
            order[:len(head)] = sorted(head, key=lambda i: -rerank_scores[i])
        result = catalog.results(query, order, scores, ranked=True)
        result['failedModels'] = failed
        if rerank_name:
            for item in result['items']:
                idx = catalog.index[item['imageID']]
                if idx in rerank_scores:
                    item['rerankScore'] = float(rerank_scores[idx])
            result.update(reranked=True, rerankModel=rerank_name)
        return result
