"""Create or resume one model's corpus embeddings on a CUDA host."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import atomic_json, digest_file, validate_vectors
from model_registry import CONFIG, EMBEDDERS, MODELS, fingerprint, load_model


def embed(artifacts, data, output, name, gpu):
    rows = pq.read_table(artifacts / 'images.parquet').to_pylist()
    ids = [row['imageID'] for row in rows]
    if len(ids) != CONFIG['expected_images'] or len(set(ids)) != len(ids):
        raise ValueError('Expected 5,000 unique images')
    if ids != json.loads((artifacts / 'embedding_ids.json').read_text(encoding='utf-8')):
        raise ValueError('The image manifest and ID list disagree')

    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    dimension = MODELS[name]['dimension']
    signature = {'model': name, 'fingerprint': fingerprint(name), 'dimension': dimension,
                 'source': digest_file(artifacts / 'images.parquet')}
    state = target / 'work.json'
    if state.is_file() and json.loads(state.read_text(encoding='utf-8')) != signature:
        raise ValueError(f'Checkpoint configuration changed in {target}')
    atomic_json(state, signature)

    model = None
    chunks = []
    data = data.resolve()
    for start in range(0, len(ids), 50):
        block = rows[start:start+50]
        checkpoint = target / f'{start:05}.npz'
        expected_ids = ids[start:start+len(block)]
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if saved['ids'].tolist() != expected_ids:
                    raise ValueError(f'Checkpoint image IDs changed: {checkpoint}')
                vectors = saved['vectors']
        else:
            if model is None:
                model = load_model(name, gpu)
            paths = []
            for row in block:
                candidates = [(data / row[key]).resolve() for key in ('original_path', 'path')]
                path = next((p for p in candidates if p.is_relative_to(data) and p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(row['imageID'])
                paths.append(path)
            batch = 1 if MODELS[name]['kind'] == 'qwen' else 8
            vectors = np.concatenate([model.encode([{'image': str(path)} for path in paths[i:i+batch]])
                                      for i in range(0, len(paths), batch)])
            validate_vectors(vectors, len(block), dimension)
            temporary = checkpoint.with_suffix('.tmp.npz')
            np.savez(temporary, ids=expected_ids, vectors=vectors)
            temporary.replace(checkpoint)
            print(f'{name}: {start+len(block)}/{len(rows)}', flush=True)
        validate_vectors(vectors, len(block), dimension)
        chunks.append(vectors)

    temporary = target / 'vectors.tmp.npy'
    vectors_file = target / 'vectors.npy'
    np.save(temporary, np.concatenate(chunks).astype(np.float32, copy=False))
    temporary.replace(vectors_file)
    ids_file = target / 'ids.json'
    atomic_json(ids_file, ids)
    atomic_json(target / 'manifest.json', {
        **signature, 'count': len(ids), 'vectors_sha256': digest_file(vectors_file),
        'ids_sha256': digest_file(ids_file),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=EMBEDDERS, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    embed(args.artifacts, args.data, args.output, args.model, args.gpu)
