"""Exercise DRES protocol and credential boundaries against a mock HTTP server."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dres import install_dres


class DRESTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.submit_status=202
        self.real_client=httpx.Client
        def handler(request):
            self.calls.append(request)
            path=request.url.path
            if path.endswith('/login'):
                return httpx.Response(200,json={'username':'tester','sessionId':'private-upstream-token'})
            if path.endswith('/user'):
                return httpx.Response(200,json={'username':'tester'})
            if path.endswith('/list'):
                return httpx.Response(200,json=[{'id':'run1','name':'Practice run','status':'ACTIVE'}])
            if '/submit/' in path:
                return httpx.Response(self.submit_status,json={'status':True,'submission':'INDETERMINATE','description':'Waiting'})
            return httpx.Response(404,json={})
        self.transport=httpx.MockTransport(handler)
        app=FastAPI()
        install_dres(app,lambda:SimpleNamespace(index={'img':0}))
        # Construct TestClient before replacing upstream HTTP client.
        self.client=TestClient(app)
        self.mock=patch('dres.httpx.Client',side_effect=lambda **kw:self.real_client(transport=self.transport,**kw))
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.client.close()

    def login(self):
        return self.client.post('/api/images/dres/login',json={'server':'https://dres.test',
            'username':'tester','password':'private-password'})

    def test_login_select_and_image_submission_use_v2_without_time_guess(self):
        response=self.login()
        self.assertEqual(response.status_code,200)
        self.assertNotIn('private-upstream-token',response.text)
        self.assertNotIn('private-password',response.text)
        self.assertIn('HttpOnly',response.headers['set-cookie'])
        self.assertIn('SameSite=strict',response.headers['set-cookie'])
        self.assertEqual(json.loads(self.calls[0].content),{'username':'tester','password':'private-password'})
        self.assertEqual(self.client.post('/api/images/dres/evaluation',json={'evaluation_id':'run1'}).status_code,200)
        result=self.client.post('/api/images/dres/submit',json={'image_id':'img','media_item_name':'img.jpg','collection':'Kitchen','task_id':'task1'})
        self.assertEqual(result.json()['http_status'],202)
        self.assertEqual(result.json()['submission'],'INDETERMINATE')
        request=self.calls[-1]
        self.assertEqual(request.url.path,'/api/v2/submit/run1')
        self.assertEqual(request.url.params['session'],'private-upstream-token')
        self.assertEqual(json.loads(request.content),{'answerSets':[{'taskId':'task1','answers':[{'mediaItemName':'img.jpg','mediaItemCollectionName':'Kitchen'}]}]})
        self.assertNotIn('start',result.text)

    def test_session_login_logout_and_missing_run(self):
        response=self.client.post('/api/images/dres/login',json={'server':'https://dres.test','session_id':'existing-token'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.calls[0].url.path,'/api/v2/user')
        self.assertEqual(self.calls[0].url.params['session'],'existing-token')
        body={'image_id':'img','media_item_name':'img'}
        self.assertEqual(self.client.post('/api/images/dres/submit',json=body).status_code,422)
        self.assertEqual(self.client.post('/api/images/dres/evaluation',json={'evaluation_id':'unknown'}).status_code,422)
        self.client.post('/api/images/dres/logout')
        self.assertFalse(self.client.get('/api/images/dres/session').json()['connected'])
        self.assertEqual(self.client.post('/api/images/dres/submit',json=body).status_code,401)

    def test_rejection_is_not_retried_and_cross_origin_is_blocked(self):
        self.login()
        self.client.post('/api/images/dres/evaluation',json={'evaluation_id':'run1'})
        self.submit_status=412
        before=len(self.calls)
        result=self.client.post('/api/images/dres/submit',json={'image_id':'img','media_item_name':'img'})
        self.assertEqual(result.status_code,412)
        self.assertEqual(len(self.calls),before+1)
        self.assertEqual(self.client.post('/api/images/dres/logout',headers={'Origin':'https://unrelated.test'}).status_code,403)
        self.assertTrue(self.client.get('/api/images/dres/session').json()['connected'])

    def test_invalid_server_and_expired_sessions(self):
        self.assertEqual(self.client.post('/api/images/dres/login',json={'server':'https://user:secret@dres.test'}).status_code,422)
        self.login()
        with patch('dres.time.monotonic',return_value=10**20):
            self.assertFalse(self.client.get('/api/images/dres/session').json()['connected'])


if __name__=='__main__': unittest.main()
