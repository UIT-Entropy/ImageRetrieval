"""Pinned Qwen VL reranking and sequential GPU model lifecycle."""
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
import sys
import threading
from functools import lru_cache

import numpy as np

MODEL = 'Qwen/Qwen3-VL-Reranker-2B'
REVISION = '4bd860ac4f15ad1897a214615cccc700f8f71818'
INSTRUCTION = "Retrieve images relevant to the user's query."


def load_options():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for Qwen inference')
    quant = os.environ.get('QWEN_QUANTIZATION','none')
    if quant not in ('none','4bit'): raise ValueError('QWEN_QUANTIZATION must be none or 4bit')
    options={'device_map':'auto','low_cpu_mem_usage':True}
    budget=os.environ.get('QWEN_GPU_MEMORY')
    if budget:
        options['max_memory']={i:budget for i in range(torch.cuda.device_count())}
        options['max_memory']['cpu']=os.environ.get('QWEN_CPU_MEMORY','4GiB')
    options['offload_folder']=str(Path(__file__).parent/'.work'/'model-offload')
    if quant=='4bit':
        from transformers import BitsAndBytesConfig
        options['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True)
    return options


class QwenReranker:
    def __init__(self):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        path=hf_hub_download(MODEL,'scripts/qwen3_vl_reranker.py',revision=REVISION,token=False)
        spec=importlib.util.spec_from_file_location('pinned_qwen_reranker',path)
        helper=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=helper
        spec.loader.exec_module(helper)
        # The upstream constructor calls .to(cuda); avoid collapsing device_map/offload.
        lm=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,revision=REVISION,token=False,
            torch_dtype=torch.float16,attn_implementation='sdpa',**load_options()).eval()
        self.lm=lm
        self.formatter=helper.Qwen3VLReranker.__new__(helper.Qwen3VLReranker)
        f=self.formatter
        f.processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION,token=False,padding_side='left')
        f.max_length=4096; f.min_pixels=4096; f.max_pixels=131072
        f.total_pixels=262144; f.fps=1; f.num_frames=1; f.max_frames=1
        f.default_instruction=INSTRUCTION
        vocab=f.processor.tokenizer.get_vocab()
        # Only yes/no logits are needed; keep the scoring head tiny and unquantized.
        self.weight=(lm.lm_head.weight[vocab['yes']]-lm.lm_head.weight[vocab['no']]).detach().clone()

    def score(self,text,image,documents):
        import torch
        f=self.formatter
        scores=[]
        for document in documents:
            pair=f.format_mm_instruction(text or None,image,None,None,document,None,instruction=INSTRUCTION,fps=1,max_frames=1)
            inputs=f.tokenize([pair]).to(self.lm.get_input_embeddings().weight.device)
            with torch.inference_mode():
                hidden=self.lm.model(**inputs,use_cache=False).last_hidden_state[:,-1]
                value=torch.sigmoid((hidden.float()*self.weight.to(hidden.device).float()).sum(-1))
            scores.append(float(value.item()))
        return validate_scores(scores,len(documents))


def validate_scores(scores,count):
    values=np.asarray(scores,dtype=np.float32)
    if values.shape!=(count,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Invalid Qwen reranker scores')
    return values.tolist()


class GPUModels:
    """One model at a time, shared by all encoder/reranker requests."""
    def __init__(self,config):
        self.config=config
        self.lock=threading.RLock()
        self.embedding=None
        self.reranker=None

    def release(self,name):
        if getattr(self,name) is not None:
            setattr(self,name,None)
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def encode(self,items):
        with self.lock:
            self.release('reranker')
            if self.embedding is None:
                from pipeline import QwenEncoder
                self.embedding=QwenEncoder(self.config,load_options=load_options())
            return self.embedding.encode(items)

    def score(self,text,image,documents):
        with self.lock:
            self.release('embedding')
            if self.reranker is None: self.reranker=QwenReranker()
            return self.reranker.score(text,image,documents)


class QueryReranker:
    def __init__(self,catalog,gpu):
        self.catalog=catalog
        self.gpu=gpu
        self.mode=os.environ.get('IMAGE_RERANKER_MODE','disabled')

    def score(self,text,image,ids):
        return self._score(text,str(image) if image else None,tuple(ids))

    @lru_cache(maxsize=32)
    def _score(self,text,image,ids):
        image=Path(image) if image else None
        documents=[self.catalog.image_path(i) for i in ids]
        if self.mode=='local': return self.gpu.score(text,image,[str(p) for p in documents])
        if self.mode=='remote':
            import requests
            from contextlib import ExitStack
            url=os.environ.get('IMAGE_ENCODER_URL','').rstrip('/')
            key=os.environ.get('IMAGE_ENCODER_KEY','')
            if not url or not key: raise RuntimeError('Remote GPU connection missing')
            with ExitStack() as stack:
                files=[('images',(p.name,stack.enter_context(p.open('rb')),'image/jpeg')) for p in documents]
                if image: files.append(('query_image',(image.name,stack.enter_context(image.open('rb')),'image/jpeg')))
                response=requests.post(url+'/api/images/rerank',data={'text':text},files=files,
                    headers={'Authorization':'Bearer '+key},timeout=(15,300))
            response.raise_for_status()
            data=response.json()
            if data.get('model')!=MODEL or data.get('revision')!=REVISION: raise ValueError('Reranker revision mismatch')
            return validate_scores(data['scores'],len(ids))
        raise RuntimeError('Reranker disabled')


def rerank_page(catalog,query,vector,reranker,image=None):
    """Rerank a fixed first-stage prefix before pagination; keep tail order unchanged."""
    if not query.text.strip() and image is None: raise ValueError('Rerank needs query text or image')
    count=query.rerank_k
    if query.offset>=count:
        result=catalog.search(query,vector)
        result.update(reranked=True,rerankModel=MODEL)
        return result
    recall=query.model_copy(update={'offset':0,'limit':max(count,query.offset+query.limit)})
    result=catalog.search(recall,vector)
    head=result['items'][:count]
    if head:
        scores=validate_scores(reranker.score(query.text.strip(),image,[r['imageID'] for r in head]),len(head))
        for hit,score in zip(head,scores): hit['rerankScore']=score
        head.sort(key=lambda r:-r['rerankScore'])
    result['items']=(head+result['items'][count:])[query.offset:query.offset+query.limit]
    result.update(offset=query.offset,limit=query.limit,reranked=True,rerankModel=MODEL)
    return result
