"""Portable metadata, atomic checkpoints, colors and output validation (no GPU)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_manifest(data_root, expected=5000):
    root = Path(data_root)
    index = {}
    for path in sorted(root.rglob('*.jpg')):
        index.setdefault(path.name, []).append(path)
    rows, ids, used = [], set(), set()
    with (root / 'metadata_sample.csv').open(encoding='utf-8-sig', newline='') as source:
        for raw in csv.DictReader(source):
            image_id = raw['imageID']
            if image_id in ids:
                raise ValueError(f'Duplicate imageID: {image_id}')
            ids.add(image_id)
            prefix = f"{raw['day']}_{raw['owner']}_"
            if not image_id.startswith(prefix):
                raise ValueError(f'ID/metadata mismatch: {image_id}')
            name = image_id[len(prefix):] + '.jpg'
            candidates = index.get(name, [])
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one image for {image_id}, found {len(candidates)}')
            path = candidates[0]
            if path in used:
                raise ValueError(f'Image mapped twice: {name}')
            used.add(path)
            match = re.fullmatch(r'(\d+)_(\d+)\.jpg', name)
            if not match or int(match[1]) != int(raw['hour']) or int(match[2]) != int(raw['frame']):
                raise ValueError(f'Hour/frame mismatch: {image_id}')
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            classes = json.loads(raw['class'])
            counts = json.loads(raw['class_count'])
            if not isinstance(classes, list) or not all(isinstance(x, str) for x in classes):
                raise ValueError(f'Invalid classes: {image_id}')
            if not isinstance(counts, dict) or not all(isinstance(v, int) and v >= 0 for v in counts.values()):
                raise ValueError(f'Invalid counts: {image_id}')
            relative = path.relative_to(root)
            rows.append({**raw, 'hour': int(raw['hour']), 'frame': int(raw['frame']),
                         'source': raw['owner'], 'batch': relative.parts[0],
                         'path': 'images/' + image_id + '.jpg', 'original_path': relative.as_posix(),
                         'width': width, 'height': height, 'sha256': digest_file(path)})
    if len(rows) != expected or len(used) != len(index) or len(used) != sum(map(len, index.values())):
        raise ValueError(f'Incomplete dataset: rows={len(rows)}, files={sum(map(len,index.values()))}, expected={expected}')
    return sorted(rows, key=lambda r: (r['day'], r['source'], r['hour'], r['frame'], r['imageID']))


def smoke_indices(rows, count=20):
    """Round-robin hours, picking evenly spread frames within each hour."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row['day'], row['source'], row['hour']), []).append(i)
    result, round_no = [], 0
    while len(result) < min(count, len(rows)):
        changed = False
        for key in sorted(groups):
            group = groups[key]
            if round_no < len(group):
                # First choose middle, then alternate toward the edges.
                order = [len(group)//2] + [j for j in range(len(group)) if j != len(group)//2]
                result.append(group[order[round_no]])
                changed = True
                if len(result) >= min(count, len(rows)):
                    break
        if not changed:
            break
        round_no += 1
    return result


def color_distribution(rgb, mask):
    """Heuristic HSV palette over foreground only; retains all color fractions."""
    pixels = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != pixels.shape[:2] or not mask.any():
        raise ValueError('Mask must have foreground pixels and match image dimensions')
    hsv = np.asarray(Image.fromarray(pixels).convert('HSV'), dtype=np.float32)[mask]
    h, s, v = hsv[:, 0] * (360 / 255), hsv[:, 1] / 255, hsv[:, 2] / 255
    names = np.full(len(h), 'gray', dtype='<U8')
    names[(s < .18) & (v > .78)] = 'white'
    chroma = s >= .18
    for name, low, high in [('red',0,15),('orange',15,45),('yellow',45,70),('green',70,165),
                            ('cyan',165,195),('blue',195,260),('purple',260,315),('pink',315,345),('red',345,361)]:
        names[chroma & (h >= low) & (h < high)] = name
    names[chroma & (h >= 15) & (h < 50) & (v < .65)] = 'brown'
    names[v < .18] = 'black'
    unique, counts = np.unique(names, return_counts=True)
    return dict(sorted(((str(n), float(c / len(names))) for n, c in zip(unique, counts)), key=lambda x: -x[1]))


def validate_vectors(vectors, count, dim=4096):
    if vectors.shape != (count, dim):
        raise ValueError(f'Unexpected vector shape: {vectors.shape}')
    for start in range(0, count, 256):
        part = np.asarray(vectors[start:start+256], dtype=np.float32)
        if not np.isfinite(part).all() or not np.allclose(np.linalg.norm(part, axis=1), 1, atol=1e-4):
            raise ValueError(f'Invalid/non-normalized vectors near row {start}')


def read_embedding_shard(path, ids, dim):
    with np.load(path, allow_pickle=False) as shard:
        if shard['ids'].tolist() != ids:
            raise ValueError(f'Checkpoint IDs mismatch: {path}')
        vectors = shard['vectors'].astype(np.float32)
    validate_vectors(vectors, len(ids), dim)
    return vectors
