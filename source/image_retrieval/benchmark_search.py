"""Measure the multi-model search API using a shared, labeled query set."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import requests


def benchmark(url, queries):
    cases = [
        ('fast', 'off'), ('2b', 'off'), ('8b', 'off'),
        ('ensemble', 'off'), ('2b', '2b'), ('8b', '8b'),
    ]
    with requests.Session() as session:
        for mode, reranker in cases:
            timings = []
            recalls = []
            for item in queries:
                payload = {**item['query'], 'search_mode': mode, 'rerank_model': reranker,
                           'offset': 0, 'limit': 20, 'rerank_k': 20}
                started = perf_counter()
                response = session.post(url.rstrip('/') + '/api/images/search', json=payload, timeout=180)
                response.raise_for_status()
                result = response.json()
                timings.append(perf_counter() - started)
                if result.get('failedModels'):
                    raise RuntimeError(f'Models unavailable: {result["failedModels"]}')
                expected = set(item.get('image_ids', []))
                if expected:
                    actual = {hit['imageID'] for hit in result['items']}
                    recalls.append(len(expected & actual) / len(expected))
            p50, p95 = np.percentile(timings, [50, 95])
            print(json.dumps({'mode': mode, 'reranker': reranker, 'queries': len(timings),
                              'p50_s': round(float(p50), 3), 'p95_s': round(float(p95), 3),
                              'recall_at_20': float(np.mean(recalls)) if recalls else None}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queries', type=Path, help='JSON list of 20 {query, image_ids} records')
    parser.add_argument('--url', default='http://127.0.0.1:8100')
    args = parser.parse_args()
    queries = json.loads(args.queries.read_text(encoding='utf-8'))
    if not isinstance(queries, list) or len(queries) != 20:
        parser.error('Supply exactly 20 benchmark queries')
    benchmark(args.url, queries)
