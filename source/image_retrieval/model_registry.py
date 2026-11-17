"""Shared model settings for Kaggle inference and corpus indexing."""
import json
from hashlib import sha256
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
EMBED_8B = 'qwen8b'
EMBED_2B = 'qwen2b'
RERANK_8B = 'rerank8b'
RERANK_2B = 'rerank2b'
SMALL_MODELS = ('siglip2', 'eva', 'dfn')
MODELS = {
    EMBED_8B: {**CONFIG['embedding'], 'kind': 'qwen', 'precision': 'nf4'},
    EMBED_2B: {**CONFIG['embedding'], 'kind': 'qwen', 'precision': 'float16',
               'model_id': 'Qwen/Qwen3-VL-Embedding-2B',
               'revision': '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda', 'dimension': 2048},
    RERANK_8B: {'kind': 'reranker', 'precision': 'nf4',
                'model_id': 'Qwen/Qwen3-VL-Reranker-8B',
                'revision': 'b212dc8c91a8164aef1ea2de9c1a867611e75c04'},
    RERANK_2B: {'kind': 'reranker', 'precision': 'float16',
                'model_id': 'Qwen/Qwen3-VL-Reranker-2B',
                'revision': '4bd860ac4f15ad1897a214615cccc700f8f71818'},
    'siglip2': {'kind': 'siglip2', 'precision': 'float16', 'dimension': 1152,
                'model_id': 'google/siglip2-so400m-patch14-384',
                'revision': 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f'},
    'eva': {'kind': 'open_clip', 'precision': 'float16', 'dimension': 768,
            'model_id': 'timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',
            'revision': 'bf4190eb65dd5204ffb03e980108beb1200e0873',
            'checkpoint': 'open_clip_model.safetensors'},
    'dfn': {'kind': 'open_clip', 'precision': 'float16', 'dimension': 1024,
            'model_id': 'apple/DFN5B-CLIP-ViT-H-14',
            'revision': '11738501a1db6d5e0a3451a71ba100be02e577e6',
            'checkpoint': 'open_clip_pytorch_model.bin'},
}
EMBEDDERS = (EMBED_8B, EMBED_2B, *SMALL_MODELS)
WORKERS = {
    'qwen8b': {EMBED_8B: 0, RERANK_8B: 1},
    'qwen2b': {EMBED_2B: 0, RERANK_2B: 1},
    'small': {'siglip2': 0, 'eva': 0, 'dfn': 1},
    'combined': {EMBED_2B: 0, RERANK_2B: 0, **dict.fromkeys(SMALL_MODELS, 1)},
}


def fingerprint(name):
    payload = {'schema': 1, 'model': MODELS[name]}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_model(name, gpu):
    import torch

    spec = MODELS[name]
    if spec['kind'] in ('qwen', 'reranker'):
        options = {'device_map': {'': gpu}, 'low_cpu_mem_usage': True}
        if spec['precision'] == 'nf4':
            from transformers import BitsAndBytesConfig
            options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            )
        if spec['kind'] == 'reranker':
            from reranker import QwenReranker
            return QwenReranker(spec['model_id'], spec['revision'], options)
        from pipeline import QwenEncoder
        return QwenEncoder(spec, load_options=options)
    return ContrastiveEncoder(spec, f'cuda:{gpu}')


class ContrastiveEncoder:
    def __init__(self, spec, device):
        import torch

        self.device = device
        if spec['kind'] == 'siglip2':
            from transformers import AutoModel, AutoProcessor
            kwargs = {'revision': spec['revision'], 'token': False}
            processor = AutoProcessor.from_pretrained(spec['model_id'], **kwargs)
            model = AutoModel.from_pretrained(spec['model_id'], torch_dtype=torch.float16, **kwargs)
            self.model = model.to(device).eval()
            self.image_input = lambda images: processor(images=images, return_tensors='pt').pixel_values
            self.text_input = lambda texts: processor(
                text=texts, padding='max_length', truncation=True, return_tensors='pt',
            )
            self.siglip = True
        else:
            import open_clip
            from huggingface_hub import hf_hub_download
            args = {'revision': spec['revision'], 'token': False}
            config = hf_hub_download(spec['model_id'], 'open_clip_config.json', **args)
            hf_hub_download(spec['model_id'], spec['checkpoint'], **args)
            source = 'local-dir:' + str(Path(config).parent)
            self.model, _, transform = open_clip.create_model_and_transforms(
                source, precision='fp16', device=device, require_pretrained=True,
            )
            self.model.eval()
            self.image_input = lambda images: torch.stack([transform(image) for image in images])
            self.text_input = open_clip.get_tokenizer(source)
            self.siglip = False

    def encode(self, items):
        import torch
        from PIL import Image, ImageOps

        normalize = torch.nn.functional.normalize
        vectors = [None] * len(items)
        for modality in ('image', 'text'):
            positions = [i for i, item in enumerate(items) if item.get(modality) is not None]
            if not positions:
                continue
            values = [items[i][modality] for i in positions]
            with torch.inference_mode():
                if modality == 'image':
                    images = []
                    for value in values:
                        if isinstance(value, Image.Image):
                            images.append(value)
                        else:
                            with Image.open(value) as opened:
                                images.append(ImageOps.exif_transpose(opened).convert('RGB'))
                    batch = self.image_input(images).to(self.device, dtype=torch.float16)
                    features = self.model.get_image_features(pixel_values=batch) if self.siglip else self.model.encode_image(batch)
                else:
                    batch = self.text_input(values).to(self.device)
                    features = self.model.get_text_features(**batch) if self.siglip else self.model.encode_text(batch)
                for i, vector in zip(positions, normalize(features.float(), dim=-1)):
                    vectors[i] = vector if vectors[i] is None else vectors[i] + vector
        if any(vector is None for vector in vectors):
            raise ValueError('Embedding requires text or an image')
        return normalize(torch.stack(vectors), dim=-1).cpu().numpy()
