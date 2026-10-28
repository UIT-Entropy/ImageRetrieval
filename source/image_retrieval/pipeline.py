"""Kaggle GPU worker. Public checkpoints only; no captioning or API credentials."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from common import (atomic_json, color_distribution, digest_file, fingerprint,
                    read_embedding_shard, smoke_indices, validate_vectors)

ACTIVE_IMAGE = None


def write_parquet(path, rows, schema=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    tmp = path.with_suffix('.tmp.parquet')
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp)
    os.replace(tmp, path)


def object_schema():
    import pyarrow as pa
    return pa.schema([
        ('imageID', pa.string()), ('instance_id', pa.string()), ('label', pa.string()),
        ('confidence', pa.float32()), ('bbox_xyxy', pa.list_(pa.float32())),
        ('mask_path', pa.string()), ('mask_sha256', pa.string()),
        ('mask_quality', pa.float32()), ('mask_area', pa.int64()),
        ('dominant_color', pa.string()), ('colors_json', pa.string()),
    ])


def load_rows(root):
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line]
    for row in rows:
        relative = Path(row['path'])
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
            raise ValueError('Unsafe image path in manifest')
    if len({r['imageID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    return rows


def gpu_report():
    import torch
    return [{'name': torch.cuda.get_device_name(i),
             'total_gib': round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
             'peak_allocated_gib': round(torch.cuda.max_memory_allocated(i) / 2**30, 3),
             'peak_reserved_gib': round(torch.cuda.max_memory_reserved(i) / 2**30, 3)}
            for i in range(torch.cuda.device_count())]


class QwenEncoder:
    """Official Qwen embedding wrapper with explicit multi-GPU loading.

    The official wrapper calls .to(cuda) after from_pretrained, which would
    collapse a sharded model. Instantiate its component class directly instead.
    Query code must reuse this class, the pinned config and its instruction.
    """
    def __init__(self, cfg, load_options=None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor
        path = hf_hub_download(cfg['model_id'], 'scripts/qwen3_vl_embedding.py', revision=cfg['revision'], token=False)
        spec = importlib.util.spec_from_file_location('pinned_qwen_embedding', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.helper_sha256 = digest_file(path)
        self.cfg = cfg
        self.helper = module
        options = load_options or {'device_map':'balanced','max_memory':{0:'13GiB',1:'13GiB'}}
        self.model = module.Qwen3VLForEmbedding.from_pretrained(
            cfg['model_id'], revision=cfg['revision'], token=False,
            torch_dtype=torch.float16, attn_implementation='sdpa', **options,
        ).eval()
        device_map = self.model.hf_device_map
        devices = {str(x) for x in device_map.values()}
        if load_options is None and not {'0', '1'}.issubset(devices) and not {'cuda:0', 'cuda:1'}.issubset(devices):
            raise RuntimeError(f'Expected sharding on two GPUs, got {device_map}')
        if load_options is None and devices & {'cpu', 'disk'}:
            raise RuntimeError('Model does not fit on two GPUs without offload')
        self.processor = AutoProcessor.from_pretrained(cfg['model_id'], revision=cfg['revision'], token=False, padding_side='right')
        # Formatting only; do not invoke the official single-GPU constructor.
        self.formatter = module.Qwen3VLEmbedder.__new__(module.Qwen3VLEmbedder)
        for key in ('min_pixels', 'max_pixels', 'max_length'):
            setattr(self.formatter, key, cfg[key])
        self.formatter.default_instruction = cfg['instruction']
        self.formatter.total_pixels = cfg['max_pixels'] * 10
        self.formatter.fps = 1
        self.formatter.max_frames = 64

    def encode(self, items):
        import torch
        from qwen_vl_utils import process_vision_info
        conversations = [self.formatter.format_model_input(**item) for item in items]
        texts = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        # Raise on preprocessing failure; never silently embed a NULL document.
        images, videos = process_vision_info(conversations, image_patch_size=16)
        inputs = self.processor(text=texts, images=images, videos=videos, padding=True,
                                truncation=True, max_length=self.cfg['max_length'],
                                do_resize=False, return_tensors='pt')
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False)
            vector = self.helper.Qwen3VLEmbedder._pooling_last(
                outputs.last_hidden_state, inputs['attention_mask'].to(outputs.last_hidden_state.device))
            vector = torch.nn.functional.normalize(vector.float(), p=2, dim=-1)
        result = vector.cpu().numpy()
        validate_vectors(result, len(items), self.cfg['dimension'])
        return result


def embedding_worker(root, out, rows, config, smoke):
    global ACTIVE_IMAGE
    cfg = config['embedding']
    shard_dir = out / 'checkpoints' / 'embeddings'
    shard_dir.mkdir(parents=True, exist_ok=True)
    indices = smoke_indices(rows, config['smoke_images']) if smoke else list(range(len(rows)))
    pending = []
    # Per-image smoke shards are also reused by the full pass.
    for i in indices:
        path = shard_dir / f'{i:06d}.npz'
        if path.exists():
            read_embedding_shard(path, [rows[i]['imageID']], cfg['dimension'])
        else:
            pending.append(i)
    start = time.monotonic()
    encoder = None
    if pending or smoke:
        encoder = QwenEncoder(cfg)
        atomic_json(out / 'embedding_runtime.json', {
            **cfg, 'helper_sha256': encoder.helper_sha256,
            'device_map': {k: str(v) for k, v in encoder.model.hf_device_map.items()},
            'pooling': 'last_non_padding_token', 'normalization': 'L2', 'output_dtype': 'float32',
        })
    for done, i in enumerate(pending, 1):
        ACTIVE_IMAGE = rows[i]['imageID']
        vector = encoder.encode([{'image': str((root / rows[i]['path']).resolve())}])
        path = shard_dir / f'{i:06d}.npz'
        temp = path.with_suffix('.tmp.npz')
        np.savez(temp, ids=np.array([rows[i]['imageID']]), vectors=vector)
        os.replace(temp, path)
        if done % cfg['checkpoint_size'] == 0 or done == len(pending):
            print(f'Embedding: {done}/{len(pending)} newly processed', flush=True)
    ACTIVE_IMAGE = None
    if smoke:
        queries = ['A person washing a cup at the kitchen sink.', 'People sitting around a dining table.']
        query_vectors = encoder.encode([{'text': text} for text in queries])
        sample = np.concatenate([read_embedding_shard(shard_dir / f'{i:06d}.npz', [rows[i]['imageID']], cfg['dimension']) for i in indices])
        scores = query_vectors @ sample.T
        atomic_json(out / 'smoke_queries.json', [
            {'query': query, 'results': [{'imageID': rows[indices[j]]['imageID'], 'score': float(scores[q, j])}
                                        for j in np.argsort(-scores[q])[:5]]}
            for q, query in enumerate(queries)
        ])
    else:
        temp = out / 'image_embeddings.tmp.npy'
        vectors = np.lib.format.open_memmap(temp, mode='w+', dtype='float32', shape=(len(rows), cfg['dimension']))
        for i, row in enumerate(rows):
            vectors[i] = read_embedding_shard(shard_dir / f'{i:06d}.npz', [row['imageID']], cfg['dimension'])[0]
        vectors.flush()
        validate_vectors(vectors, len(rows), cfg['dimension'])
        del vectors
        os.replace(temp, out / 'image_embeddings.npy')
        atomic_json(out / 'embedding_ids.json', [r['imageID'] for r in rows])
    atomic_json(out / ('smoke_embedding.json' if smoke else 'embedding_metrics.json'),
                {'images': len(indices), 'new_images': len(pending), 'seconds': time.monotonic()-start, 'gpus': gpu_report()})


class ObjectEncoder:
    def __init__(self, config):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model
        self.cfg = config['detection']
        dc, sc = config['detection'], config['segmentation']
        self.det_processor = AutoProcessor.from_pretrained(dc['model_id'], revision=dc['revision'], token=False)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(dc['model_id'], revision=dc['revision'], token=False).to('cuda:0').eval()
        self.seg_processor = Sam2Processor.from_pretrained(sc['model_id'], revision=sc['revision'], token=False)
        self.segmenter = Sam2Model.from_pretrained(sc['model_id'], revision=sc['revision'], token=False).to('cuda:1').eval()

    def detect(self, image, labels):
        import torch
        from torchvision.ops import batched_nms
        detections = []
        n = self.cfg['labels_per_prompt']
        for start in range(0, len(labels), n):
            chunk = labels[start:start+n]
            inputs = self.det_processor(images=image, text='. '.join(chunk) + '.', return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                outputs = self.detector(**inputs)
            results = self.det_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.cfg['box_threshold'],
                text_threshold=self.cfg['text_threshold'], target_sizes=[image.size[::-1]])[0]
            for box, score, label in zip(results['boxes'].cpu().tolist(), results['scores'].cpu().tolist(), results['text_labels']):
                label = label.strip().lower().rstrip('.')
                # Phrase decoding can concatenate labels: do not invent a class.
                if label in chunk and box[2] > box[0] and box[3] > box[1]:
                    detections.append({'bbox_xyxy': box, 'confidence': score, 'label': label})
        if not detections:
            return []
        boxes = torch.tensor([x['bbox_xyxy'] for x in detections])
        scores = torch.tensor([x['confidence'] for x in detections])
        label_ids = torch.tensor([labels.index(x['label']) for x in detections])
        keep = batched_nms(boxes, scores, label_ids, self.cfg['nms_iou']).tolist()
        return [detections[i] for i in keep]

    def masks(self, image, detections):
        import torch
        n = self.cfg['mask_batch_size']
        for start in range(0, len(detections), n):
            chunk = detections[start:start+n]
            inputs = self.seg_processor(images=image, input_boxes=[[x['bbox_xyxy'] for x in chunk]], return_tensors='pt').to('cuda:1')
            with torch.inference_mode():
                output = self.segmenter(**inputs, multimask_output=False)
            masks = self.seg_processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
            qualities = output.iou_scores.detach().cpu().reshape(-1).tolist()
            for detection, mask, quality in zip(chunk, masks, qualities):
                yield detection, mask.squeeze(0).numpy().astype(bool), quality


def read_object_checkpoint(path, row, out):
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['imageID'] != row['imageID'] or result['image_sha256'] != row['sha256']:
        raise ValueError(f'Object checkpoint mismatch: {path}')
    for obj in result['objects']:
        mask_path = (out / obj['mask_path']).resolve()
        if not mask_path.is_relative_to(out.resolve()) or not mask_path.is_file() or digest_file(mask_path) != obj['mask_sha256']:
            raise ValueError(f'Missing/corrupt mask in checkpoint: {path}')
    return result['objects']


def object_worker(root, out, rows, config, smoke):
    global ACTIVE_IMAGE
    indices = smoke_indices(rows, config['smoke_images']) if smoke else list(range(len(rows)))
    checkpoints = out / 'checkpoints' / 'objects'
    checkpoints.mkdir(parents=True, exist_ok=True)
    masks_dir = out / 'masks'
    masks_dir.mkdir(exist_ok=True)
    pending = []
    for i in indices:
        path = checkpoints / f'{i:06d}.json'
        if path.exists():
            read_object_checkpoint(path, rows[i], out)
        else:
            pending.append(i)
    start, object_count = time.monotonic(), 0
    encoder = ObjectEncoder(config) if pending else None
    for done, i in enumerate(pending, 1):
        row = rows[i]
        ACTIVE_IMAGE = row['imageID']
        with Image.open(root / row['path']) as opened:
            image = opened.convert('RGB')
        detections = encoder.detect(image, config['labels'])
        objects = []
        for j, (det, mask, quality) in enumerate(encoder.masks(image, detections)):
            if not mask.any():
                continue
            instance = f"{row['imageID']}__{j:03d}"
            mask_path = masks_dir / f'{instance}.png'
            Image.fromarray(mask.astype('uint8') * 255).save(mask_path)
            colors = color_distribution(image, mask)
            objects.append({**det, 'imageID': row['imageID'], 'instance_id': instance,
                            'mask_path': mask_path.relative_to(out).as_posix(), 'mask_sha256': digest_file(mask_path),
                            'mask_quality': float(quality), 'mask_area': int(mask.sum()),
                            'dominant_color': next(iter(colors)), 'colors_json': json.dumps(colors)})
        atomic_json(checkpoints / f'{i:06d}.json', {'imageID': row['imageID'], 'image_sha256': row['sha256'], 'objects': objects})
        if done % 10 == 0 or done == len(pending):
            print(f'Object/color: {done}/{len(pending)} newly processed', flush=True)
    ACTIVE_IMAGE = None
    all_objects = []
    for i in indices:
        objects = read_object_checkpoint(checkpoints / f'{i:06d}.json', rows[i], out)
        object_count += len(objects)
        if not smoke:
            all_objects.extend(objects)
        else:
            make_overlay(root, out, rows[i], objects)
    if smoke and object_count == 0:
        raise RuntimeError('Smoke test detected zero objects; inspect vocabulary and thresholds')
    if not smoke:
        write_parquet(out / 'objects.parquet', all_objects, object_schema())
    atomic_json(out / ('smoke_objects.json' if smoke else 'object_metrics.json'),
                {'images': len(indices), 'new_images': len(pending), 'objects': object_count,
                 'seconds': time.monotonic()-start, 'gpus': gpu_report()})


def make_overlay(root, out, row, objects):
    with Image.open(root / row['path']) as image:
        image = image.convert('RGB')
    pixels = np.asarray(image).copy()
    for obj in objects:
        with Image.open(out / obj['mask_path']) as mask_image:
            mask = np.asarray(mask_image) > 0
        pixels[mask] = (pixels[mask].astype(float) * .7 + np.array([0, 255, 100]) * .3).astype('uint8')
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for obj in objects:
        draw.rectangle(obj['bbox_xyxy'], outline='red', width=2)
        draw.text(tuple(obj['bbox_xyxy'][:2]), f"{obj['label']} {obj['dominant_color']} {obj['confidence']:.2f}", fill='red', stroke_width=1, stroke_fill='white')
    image.thumbnail((1000, 750))
    directory = out / 'smoke_overlays'
    directory.mkdir(exist_ok=True)
    image.save(directory / (row['imageID'] + '.jpg'))


def preflight(config):
    import torch
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('This run requires exactly 2 NVIDIA T4 GPUs; no automatic hardware/model fallback')
    for group in ('embedding', 'detection', 'segmentation'):
        cfg = config[group]
        name = 'model.safetensors.index.json' if group == 'embedding' else 'model.safetensors'
        # Explicit anonymous access: no Hugging Face secret required.
        get_hf_file_metadata(hf_hub_url(cfg['model_id'], name, revision=cfg['revision']), token=False)
    return gpu_report()


def run_all(root, out, resume=None):
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    rows = load_rows(root)
    expected = config['expected_images']
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} images, got {len(rows)}')
    code_hashes = {name: digest_file(Path(__file__).with_name(name)) for name in ('pipeline.py', 'common.py')}
    identity = fingerprint({'config': config, 'manifest': digest_file(root / 'manifest.jsonl'), 'code': code_hashes})
    out.mkdir(parents=True, exist_ok=True)
    if resume:
        previous = json.loads((resume / 'run_manifest.json').read_text(encoding='utf-8'))
        if previous.get('fingerprint') != identity:
            raise ValueError('Resume dataset/config fingerprint mismatch')
        for directory in ('checkpoints', 'masks'):
            if (resume / directory).exists():
                shutil.copytree(resume / directory, out / directory, dirs_exist_ok=True)
    existing = out / 'run_manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')).get('fingerprint') != identity:
        raise ValueError('Output directory belongs to a different dataset/config')
    manifest = {'schema_version': 1, 'status': 'preflight', 'fingerprint': identity,
                'config': config, 'code_sha256': code_hashes, 'images_total': len(rows), 'images_completed': 0,
                'images_failed': 0, 'images_pending': len(rows), 'errors': [],
                'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'packages': {name: importlib.metadata.version(name) for name in
                             ['torch', 'torchvision', 'transformers', 'accelerate', 'numpy', 'pyarrow', 'qwen-vl-utils']}}
    atomic_json(existing, manifest)
    try:
        manifest['gpus'] = preflight(config)
        for i, row in enumerate(rows):
            if digest_file(root / row['path']) != row['sha256']:
                raise ValueError(f'Image checksum mismatch: {row["imageID"]}')
            with Image.open(root / row['path']) as image:
                image.load()
            if i % 1000 == 0:
                print(f'Validated input {i}/{len(rows)}', flush=True)
        write_parquet(out / 'images.parquet', rows)
        atomic_json(out / 'config.json', config)
        shutil.copy2(root / 'manifest.jsonl', out / 'manifest.jsonl')
        for stage, smoke in [('embedding', True), ('objects', True), ('embedding', False), ('objects', False)]:
            manifest['status'] = ('smoke_' if smoke else 'full_') + stage
            atomic_json(existing, manifest)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--out', str(out), '--worker', stage]
            if smoke:
                command.append('--smoke')
            (out / 'worker_error.json').unlink(missing_ok=True)
            # Separate processes guarantee GPU memory is released between models.
            subprocess.run(command, check=True)
        validate_vectors(np.load(out / 'image_embeddings.npy', mmap_mode='r'), len(rows), config['embedding']['dimension'])
        import pyarrow.parquet as pq
        if json.loads((out / 'embedding_ids.json').read_text()) != [r['imageID'] for r in rows]:
            raise RuntimeError('Final embedding ID order mismatch')
        manifest.update(status='complete', images_completed=len(rows), images_pending=0,
                        object_instances=pq.read_metadata(out / 'objects.parquet').num_rows,
                        outputs={name: digest_file(out / name) for name in ['images.parquet', 'image_embeddings.npy', 'embedding_ids.json', 'objects.parquet']})
    except Exception as exc:
        embedding_done = len(list((out / 'checkpoints' / 'embeddings').glob('*.npz')))
        objects_done = len(list((out / 'checkpoints' / 'objects').glob('*.json')))
        completed = len(set(p.stem for p in (out / 'checkpoints' / 'embeddings').glob('*.npz')) &
                        set(p.stem for p in (out / 'checkpoints' / 'objects').glob('*.json')))
        manifest.update(status='failed', failed_stage=manifest['status'], images_completed=completed,
                        images_pending=len(rows)-completed, embedding_checkpoints=embedding_done,
                        object_checkpoints=objects_done,
                        errors=[{'type': type(exc).__name__, 'message': str(exc)}])
        if (out / 'worker_error.json').exists():
            detail = json.loads((out / 'worker_error.json').read_text(encoding='utf-8'))
            manifest['errors'].append(detail)
            manifest['images_failed'] = int(detail.get('imageID') is not None)
            manifest['images_pending'] = len(rows) - completed - manifest['images_failed']
        atomic_json(existing, manifest)
        raise
    finally:
        manifest['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json(existing, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--worker', choices=['embedding', 'objects'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        cfg = json.loads((args.root / 'config.json').read_text(encoding='utf-8'))
        rows = load_rows(args.root)
        worker = embedding_worker if args.worker == 'embedding' else object_worker
        try:
            worker(args.root, args.out, rows, cfg, args.smoke)
        except Exception as exc:
            atomic_json(args.out / 'worker_error.json', {
                'type': type(exc).__name__, 'message': str(exc), 'imageID': ACTIVE_IMAGE,
                'stage': args.worker, 'smoke': args.smoke})
            raise
    else:
        run_all(args.root, args.out, args.resume)
