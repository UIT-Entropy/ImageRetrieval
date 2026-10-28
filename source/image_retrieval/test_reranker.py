import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reranker import GPUModels, validate_scores, rerank_page
from search_api import SearchRequest
import test_search_api


class RerankerTests(unittest.TestCase):
    def setUp(self):
        fixture=test_search_api.SearchTests()
        fixture.setUp()
        self.cat=fixture.cat
        self.scores=SimpleNamespace(score=lambda text,image,ids:[{'a':.2,'b':.8,'c':.5}[i] for i in ids])

    def test_reranks_before_pagination_and_preserves_cosine_and_tail(self):
        base=np.array([.8,.5,.2]+[0.]*4093,dtype=np.float32)
        query=SearchRequest(text='kitchen',rerank=True,rerank_k=2,offset=0,limit=3)
        result=rerank_page(self.cat,query,base,self.scores)
        self.assertEqual([r['imageID'] for r in result['items']],['b','a','c'])
        self.assertAlmostEqual(result['items'][0]['rerankScore'],.8)
        self.assertLess(result['items'][0]['score'],result['items'][1]['score'])
        page=rerank_page(self.cat,query.model_copy(update={'offset':1,'limit':1}),base,self.scores)
        self.assertEqual(page['items'][0]['imageID'],'a')
        tail=rerank_page(self.cat,query.model_copy(update={'offset':2,'limit':1}),base,self.scores)
        self.assertEqual(tail['items'][0]['imageID'],'c')

    def test_filters_apply_before_reranking_and_invalid_scores_fail(self):
        query=SearchRequest(text='shirt',hour_from=9,rerank=True)
        result=rerank_page(self.cat,query,None,self.scores)
        self.assertEqual([r['imageID'] for r in result['items']],['b','c'])
        for scores in ([float('nan')],[float('inf')],[-.1],[1.1],[]):
            with self.assertRaises(ValueError): validate_scores(scores,1)
        with self.assertRaises(ValueError): rerank_page(self.cat,SearchRequest(rerank=True),None,self.scores)

    def test_gpu_switches_unload_embedding_before_loading_reranker(self):
        gpu=GPUModels({})
        events=[]
        gpu.embedding=object()
        def release(name): events.append(('release',name)); setattr(gpu,name,None)
        gpu.release=release
        with patch('reranker.QwenReranker',return_value=self.scores) as factory:
            gpu.score('query',None,['a'])
        factory.assert_called_once()
        self.assertEqual(events,[('release','embedding')])
        self.assertIsNone(gpu.embedding)


if __name__=='__main__': unittest.main()
