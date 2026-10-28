"""Search invariants: instance-level color/count, ranking, time and API errors."""
from collections import defaultdict
import unittest
import os
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient

from search_api import Catalog, SearchRequest, create_app, feedback_vector


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.cat = Catalog.__new__(Catalog)
        self.cat.ids = ['a', 'b', 'c']
        self.cat.index = {v:i for i,v in enumerate(self.cat.ids)}
        self.cat.rows = [{'imageID':i,'day':'day1','source':'Kitchen','hour':8+n,
                          'frame':n,'batch':'Batch1'} for n,i in enumerate(self.cat.ids)]
        self.cat.vectors = np.eye(3,4096,dtype='float32')
        self.cat.objects = defaultdict(list, {
            'a':[{'label':'shirt','colors':{'blue':1.}}, {'label':'cup','colors':{'red':1.}}],
            'b':[{'label':'shirt','colors':{'red':.8,'black':.2}}, {'label':'cup','colors':{'white':1.}}],
            'c':[{'label':'shirt','colors':{'red':1.}}, {'label':'shirt','colors':{'red':1.}}],
        })

    def test_color_and_count_belong_to_same_object(self):
        query=SearchRequest(objects=[{'label':'shirt','color':'red','min_count':2}])
        self.assertEqual([x['imageID'] for x in self.cat.search(query)['items']],['c'])
        query=SearchRequest(objects=[{'label':'shirt','color':'red'}])
        self.assertEqual([x['imageID'] for x in self.cat.search(query)['items']],['b','c'])
        zero_threshold=SearchRequest(objects=[{'label':'shirt','color':'red','min_color_fraction':0}])
        self.assertEqual([x['imageID'] for x in self.cat.search(zero_threshold)['items']],['b','c'])

    def test_filters_are_and_and_hours_inclusive(self):
        query=SearchRequest(hour_from=9,hour_to=9,objects=[{'label':'shirt','color':'red'},{'label':'cup'}])
        self.assertEqual([x['imageID'] for x in self.cat.search(query)['items']],['b'])
        with self.assertRaises(ValueError): SearchRequest(hour_from=10,hour_to=8)

    def test_rank_then_paginate_and_empty_shortlist(self):
        result=self.cat.search(SearchRequest(offset=0,limit=1),self.cat.vectors[2])
        self.assertEqual(result['items'][0]['imageID'],'c')
        self.assertAlmostEqual(result['items'][0]['score'],1.)
        self.assertEqual(result['total'],3)
        self.assertEqual(self.cat.search(SearchRequest(shortlist_ids=[]))['total'],0)
        with self.assertRaises(ValueError): self.cat.search(SearchRequest(),np.zeros(4096))

    def test_exact_maximum_range_and_absence_counts(self):
        exact=self.cat.search(SearchRequest(objects=[{'label':'shirt','min_count':1,'max_count':1}]))
        self.assertEqual([x['imageID'] for x in exact['items']],['a','b'])
        maximum=self.cat.search(SearchRequest(objects=[{'label':'cup','min_count':0,'max_count':0}]))
        self.assertEqual([x['imageID'] for x in maximum['items']],['c'])
        ranged=self.cat.search(SearchRequest(objects=[{'label':'shirt','min_count':1,'max_count':2}]))
        self.assertEqual(ranged['total'],3)
        self.assertEqual(ranged['items'][2]['constraintCounts'][0]['count'],2)
        with self.assertRaises(ValueError): SearchRequest(objects=[{'label':'shirt','min_count':3,'max_count':1}])

    def test_original_metadata_is_separate_and_missing_is_unknown(self):
        self.cat.metadata_counts={'a':{'shirt':2},'b':{'shirt':1},'c':{}}
        result=self.cat.search(SearchRequest(objects=[{'label':'shirt','min_count':2,'max_count':2,'count_source':'metadata'}]))
        self.assertEqual([x['imageID'] for x in result['items']],['a'])
        self.assertEqual(result['items'][0]['objectCounts']['shirt'],1)
        self.assertEqual(result['items'][0]['metadataObjectCounts']['shirt'],2)
        missing=self.cat.search(SearchRequest(objects=[{'label':'shirt','min_count':0,'max_count':0,'count_source':'metadata'}]))
        self.assertEqual(missing['total'],0)
        with self.assertRaises(ValueError): SearchRequest(objects=[{'label':'shirt','color':'red','count_source':'metadata'}])

    def test_feedback_reranks_by_vectors_and_keeps_hard_filters(self):
        base=(self.cat.vectors[0]+self.cat.vectors[1])/np.sqrt(2)
        query=SearchRequest(positive_ids=['b'],negative_ids=['a'])
        vector=feedback_vector(self.cat,query,base)
        self.assertAlmostEqual(float(np.linalg.norm(vector)),1.,places=6)
        self.assertGreater(float(vector[1]),float(vector[0]))
        self.assertEqual(self.cat.search(query,vector)['items'][0]['imageID'],'b')
        query=SearchRequest(positive_ids=['b'],hour_from=10)
        result=self.cat.search(query,feedback_vector(self.cat,query,None))
        self.assertEqual([r['imageID'] for r in result['items']],['c'])
        # Duplicate IDs do not increase an image's weight.
        np.testing.assert_allclose(feedback_vector(self.cat,SearchRequest(positive_ids=['b','b']),base),feedback_vector(self.cat,SearchRequest(positive_ids=['b']),base))
        with self.assertRaises(ValueError): SearchRequest(positive_ids=['a'],negative_ids=['a'])


class RealArtifactsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        if not (Path(os.environ.get('IMAGE_ARTIFACT_DIR',Path(__file__).parent/'output'/'image_retrieval_output'))/'run_manifest.json').is_file():
            raise unittest.SkipTest('Real image artifacts are not installed')
        cls.env = patch.dict(os.environ, {'IMAGE_ENCODER_MODE':'disabled'})
        cls.env.__enter__()
        cls.context=TestClient(create_app())
        cls.client=cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None,None,None)
        cls.env.__exit__(None,None,None)

    def test_real_catalog_and_reference_ranking(self):
        self.assertEqual(self.client.get('/api/images/health').json()['images'],5000)
        first=self.client.post('/api/images/search',json={'limit':1}).json()['items'][0]['imageID']
        result=self.client.post('/api/images/search',json={'reference_id':first,'limit':3}).json()
        self.assertAlmostEqual(result['items'][0]['score'],1.,places=5)
        self.assertEqual(result['total'],5000)
        detail=self.client.get('/api/images/detail/'+first).json()
        self.assertGreater(len(detail['objects']),0)
        self.assertLessEqual(len(detail['neighbors']),9)
        self.assertEqual(self.client.get(detail['thumbnailUrl']).status_code,200)
        instance=detail['objects'][0]['instance_id']
        self.assertEqual(self.client.get('/api/images/mask/'+instance).status_code,200)

    def test_invalid_queries_fail_without_fake_scores(self):
        self.assertEqual(self.client.post('/api/images/search',json={'reference_id':'missing'}).status_code,404)
        self.assertEqual(self.client.post('/api/images/search',json={'hour_from':20,'hour_to':8}).status_code,422)
        self.assertEqual(self.client.post('/api/images/search',json={'text':'person'}).status_code,503)
        self.assertEqual(self.client.post('/api/images/query-assets',files={'image':('bad.jpg',b'not-an-image')}).status_code,400)

    def test_video_boundaries_do_not_cross_hour_and_frame_order_is_numeric(self):
        cat=self.context.app.state.catalog
        self.assertEqual(len(cat.groups),13)
        for key, ids in cat.groups.items():
            for image_id in (ids[0],ids[-1]):
                detail=self.client.get('/api/images/detail/'+image_id).json()
                self.assertTrue(all((n['day'],n['source'],n['hour'])==key for n in detail['neighbors']))
            first=self.client.get('/api/images/detail/'+ids[0]).json()
            last=self.client.get('/api/images/detail/'+ids[-1]).json()
            self.assertIsNone(first['previousImageID'])
            self.assertIsNone(last['nextImageID'])
            video=self.client.get('/api/images/video/'+ids[0]).json()
            self.assertEqual(len(video['items']),len(ids))
            frames=[r['frame'] for r in video['items']]
            self.assertEqual(frames,sorted(frames))

    def test_rerank_api_uses_fixed_recall_prefix_and_keeps_pagination(self):
        first=self.client.post('/api/images/search',json={'limit':1}).json()['items'][0]['imageID']
        body={'reference_id':first,'limit':3}
        recall=self.client.post('/api/images/search',json=body).json()
        with patch.object(self.client.app.state.reranker,'mode','local'),patch.object(self.client.app.state.reranker,'score',return_value=[.1,.9,.5]) as score:
            result=self.client.post('/api/images/search',json={**body,'rerank':True,'rerank_k':3}).json()
            page=self.client.post('/api/images/search',json={**body,'rerank':True,'rerank_k':3,'offset':1,'limit':1}).json()
        self.assertEqual(result['items'][0]['imageID'],recall['items'][1]['imageID'])
        self.assertEqual(page['items'][0]['imageID'],result['items'][1]['imageID'])
        self.assertEqual(result['total'],5000)
        self.assertTrue(result['reranked'])
        self.assertEqual(score.call_args.args[2],[r['imageID'] for r in recall['items']])


if __name__=='__main__': unittest.main()
