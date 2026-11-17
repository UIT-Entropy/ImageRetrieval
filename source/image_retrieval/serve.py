"""Run the local API with laptop GPU or a remote Kaggle GPU service."""
import argparse
import os
from pathlib import Path
import time

from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[2]


def configure(profile):
    load_dotenv(ROOT/'.env',override=False)
    if profile in ('multi', 'local') and os.environ.get('IMAGE_MULTI_INDEX'):
        if not os.environ.get('IMAGE_MULTI_INDEX'):
            raise RuntimeError('Set IMAGE_MULTI_INDEX to the directory created by build_index.py')
        os.environ['IMAGE_RUNTIME_MODE'] = 'local' if profile == 'local' else 'remote'
        if profile == 'local':
            os.environ['IMAGE_ENCODER_MODE'] = 'disabled'
            os.environ['IMAGE_RERANKER_MODE'] = 'disabled'
            return
        return
    if profile == 'local':
        raise RuntimeError('Local mode uses the multi-model index. Set IMAGE_MULTI_INDEX to the directory created by build_index.py')
    if profile == 'multi':
        raise RuntimeError('Set IMAGE_MULTI_INDEX to the directory created by build_index.py')
    mode='local' if profile=='local' else 'remote'
    os.environ['IMAGE_ENCODER_MODE']=mode
    os.environ['IMAGE_RERANKER_MODE']=mode
    if mode=='local':
        os.environ.setdefault('QWEN_QUANTIZATION','4bit')
        os.environ.setdefault('QWEN_GPU_MEMORY','4GiB')
        os.environ.setdefault('QWEN_CPU_MEMORY','4GiB')
    elif not os.environ.get('IMAGE_ENCODER_URL') or not os.environ.get('IMAGE_ENCODER_KEY'):
        raise RuntimeError('Run manage_encoder.py push / connect before selecting Kaggle')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['local','kaggle','multi'],required=True)
    parser.add_argument('--smoke',action='store_true',help='Encode one query and rerank one real image; do not start API')
    args=parser.parse_args()
    configure(args.profile)
    if args.smoke and args.profile == 'multi':
        parser.error('Start the Kaggle multi API and benchmark the search endpoint')
    if args.smoke and args.profile == 'local':
        from local_runtime import LocalModelRuntime
        runtime=LocalModelRuntime()
        model=os.environ.get('IMAGE_LOCAL_SMOKE_MODEL','siglip2')
        try:
            result=runtime.execute(model,'encode','A person in a kitchen.',None)
            print({'model':model,'embedding_dimension':len(result['vector']),'status':'ok'})
        finally:
            runtime.close()
        return
    if args.smoke:
        from search_api import Catalog, QueryEncoder, HERE
        from reranker import GPUModels, QueryReranker
        cat=Catalog(Path(os.environ.get('IMAGE_ARTIFACT_DIR',HERE/'output'/'image_retrieval_output')),
                    Path(os.environ.get('IMAGE_DATA_DIR',ROOT/'data')))
        gpu=GPUModels(cat.config['embedding'])
        started=time.monotonic()
        vector=QueryEncoder(cat,gpu).encode('A person in a kitchen.')
        scores=QueryReranker(cat,gpu).score('A person in a kitchen.',None,[cat.ids[0]])
        print({'embedding_shape':list(vector.shape),'rerank_scores':scores,'elapsed_seconds':round(time.monotonic()-started,2)})
        if args.profile=='local':
            import torch
            print({'gpu':torch.cuda.get_device_name(0),'peak_vram_gib':round(torch.cuda.max_memory_allocated()/1024**3,2)})
        return
    import uvicorn
    from search_api import create_app
    uvicorn.run(create_app(),host=os.environ.get('IMAGE_API_HOST','127.0.0.1'),port=int(os.environ.get('IMAGE_API_PORT','8100')))


if __name__=='__main__': main()
