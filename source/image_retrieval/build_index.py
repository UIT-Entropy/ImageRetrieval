"""Build an isolated SQLite catalog and Qdrant index from validated artifacts."""
import argparse
from contextlib import closing
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq

from common import atomic_json, digest_file, validate_vectors
from model_registry import CONFIG, EMBEDDERS, MODELS, fingerprint

COLLECTION = 'images'


def build(artifacts, embeddings, output):
    from qdrant_client import QdrantClient, models

    source = artifacts / 'images.parquet'
    source_digest = digest_file(source)
    rows = pq.read_table(source).to_pylist()
    identifiers = [row['imageID'] for row in rows]
    positions = {image_id: i for i, image_id in enumerate(identifiers)}
    if len(positions) != len(rows) or len(rows) != CONFIG['expected_images']:
        raise ValueError('Incomplete image metadata or duplicate image IDs')
    manifest = json.loads((artifacts / 'run_manifest.json').read_text(encoding='utf-8'))
    objects_path = artifacts / 'objects.parquet'
    for path in (source, objects_path):
        if manifest['outputs'].get(path.name) != digest_file(path):
            raise ValueError(f'Invalid metadata checksum: {path.name}')
    vectors = {}
    for name in EMBEDDERS:
        directory = embeddings / name
        metadata = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        values, ids = directory / 'vectors.npy', directory / 'ids.json'
        if (metadata['fingerprint'] != fingerprint(name) or metadata['source'] != source_digest
                or metadata['vectors_sha256'] != digest_file(values) or metadata['ids_sha256'] != digest_file(ids)
                or json.loads(ids.read_text(encoding='utf-8')) != identifiers):
            raise ValueError(f'Incompatible embeddings: {name}')
        vectors[name] = np.load(values, mmap_mode='r', allow_pickle=False)
        validate_vectors(vectors[name], len(rows), MODELS[name]['dimension'])

    generation = uuid4().hex
    target = output / generation
    target.mkdir(parents=True)
    with closing(sqlite3.connect(target / 'catalog.sqlite3')) as db, db:
        db.executescript("""
            CREATE TABLE images (
                point_id INTEGER PRIMARY KEY, image_id TEXT NOT NULL UNIQUE,
                day TEXT, source TEXT, hour INTEGER, frame INTEGER, metadata TEXT NOT NULL
            );
            CREATE INDEX images_time ON images(day, source, hour, frame);
            CREATE TABLE objects (
                image_id TEXT NOT NULL REFERENCES images(image_id), metadata TEXT NOT NULL
            );
            CREATE INDEX objects_image ON objects(image_id);
        """)
        db.execute('PRAGMA foreign_keys=ON')
        db.executemany('INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)', (
            (i, row['imageID'], row['day'], row['source'], row['hour'], row['frame'], json.dumps(row))
            for i, row in enumerate(rows)
        ))
        db.executemany('INSERT INTO objects VALUES (?, ?)', (
            (row['imageID'], json.dumps(row)) for row in pq.read_table(objects_path).to_pylist()
        ))
    client = QdrantClient(path=str(target / 'qdrant'))
    try:
        client.create_collection(COLLECTION, vectors_config={
            name: models.VectorParams(size=MODELS[name]['dimension'], distance=models.Distance.COSINE)
            for name in EMBEDDERS
        })
        for start in range(0, len(rows), 64):
            client.upsert(COLLECTION, points=[
                models.PointStruct(id=i, vector={name: values[i].tolist() for name, values in vectors.items()})
                for i in range(start, min(start+64, len(rows)))
            ], wait=True)
        if client.count(COLLECTION, exact=True).count != len(rows):
            raise RuntimeError('Qdrant index is incomplete')
    finally:
        client.close()
    atomic_json(output / 'active.json', {
        'generation': generation, 'source': source_digest,
        'models': {name: fingerprint(name) for name in EMBEDDERS},
    })
    print(f'Built index {generation}, {len(rows)} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.artifacts, args.embeddings, args.output)
