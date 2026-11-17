"""On-demand local query models for a single laptop GPU."""
from collections import OrderedDict
import gc
import os
import threading

import torch

from model_registry import EMBED_2B, EMBED_8B, MODELS, RERANK_2B, RERANK_8B, SMALL_MODELS, fingerprint, load_local_model


class LocalModelRuntime:
    def __init__(self):
        self.keep = max(1, min(3, int(os.environ.get('IMAGE_LOCAL_MODEL_CACHE', '2'))))
        self.models = OrderedDict()
        self.lock = threading.RLock()

    @staticmethod
    def supports(name):
        return name in (EMBED_2B, RERANK_2B, *SMALL_MODELS)

    def _release(self, name):
        if name in self.models:
            del self.models[name]
            gc.collect()
            torch.cuda.empty_cache()

    def _get(self, name):
        if not self.supports(name):
            raise ValueError('Local mode supports Qwen2B, SigLIP2, EVA-CLIP and DFN-CLIP; Qwen8B requires Kaggle')
        if name in self.models:
            self.models.move_to_end(name)
            return self.models[name]

        while True:
            try:
                model = load_local_model(name, gpu=0)
                self.models[name] = model
                self.models.move_to_end(name)
                while len(self.models) > self.keep:
                    self._release(next(iter(self.models)))
                return model
            except torch.cuda.OutOfMemoryError as exc:
                if not self.models:
                    raise ValueError(f'Không đủ VRAM để nạp {name}. Hãy giảm IMAGE_LOCAL_MODEL_CACHE hoặc đóng ứng dụng đang dùng GPU.') from exc
                self._release(next(iter(self.models)))

    def execute(self, name, operation, text, image, documents=()):
        with self.lock:
            try:
                model = self._get(name)
                if operation == 'encode':
                    item = {}
                    if text.strip():
                        item['text'] = text.strip()
                    if image:
                        item['image'] = str(image)
                    vector = model.encode([item])[0].tolist()
                    return {'model': name, 'fingerprint': fingerprint(name), 'vector': vector}
                if operation == 'rerank' and name in (RERANK_2B, RERANK_8B):
                    scores = model.score(text.strip(), str(image) if image else None,
                                         [str(path) for path in documents])
                    return {'model': name, 'fingerprint': fingerprint(name), 'scores': scores}
                raise ValueError('Unsupported local model operation')
            except torch.cuda.OutOfMemoryError as exc:
                for key in list(self.models):
                    if key != name:
                        self._release(key)
                gc.collect()
                torch.cuda.empty_cache()
                raise ValueError(f'Không đủ VRAM để chạy {name}. Chọn ít model hơn hoặc đóng ứng dụng đang dùng GPU.') from exc

    def close(self):
        with self.lock:
            self.models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
