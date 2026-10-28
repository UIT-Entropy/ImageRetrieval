"""DRES v2 adapter. Credentials stay in memory, never in browser storage/logs."""
from __future__ import annotations

import secrets
import threading
import time
from urllib.parse import quote, urlsplit

import httpx
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field, SecretStr

COOKIE = 'image_dres_session'
TTL = 8 * 3600


class Login(BaseModel):
    server: str = Field(min_length=1, max_length=500)
    username: str = Field(default='', max_length=200)
    password: SecretStr = Field(default=SecretStr(''), max_length=4000)
    session_id: SecretStr = Field(default=SecretStr(''), max_length=4000)


class Selection(BaseModel):
    evaluation_id: str = Field(min_length=1, max_length=200)


class Submission(BaseModel):
    image_id: str = Field(min_length=1, max_length=200)
    media_item_name: str = Field(min_length=1, max_length=300)
    collection: str = Field(default='', max_length=200)
    task_id: str = Field(default='', max_length=200)


def server_url(value):
    parts = urlsplit(value.strip())
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise HTTPException(422, 'DRES URL phải là http(s)://server, không chứa tài khoản/query.')
    return value.strip().rstrip('/')


def remote(server, method, path, token='', payload=None):
    try:
        # Never follow redirects carrying login credentials or a DRES token.
        with httpx.Client(timeout=12, follow_redirects=False, trust_env=False) as client:
            response = client.request(method, server + path,
                params={'session': token} if token else None, json=payload)
        if response.status_code not in (200, 202):
            descriptions = {400:'DRES từ chối yêu cầu hoặc task chưa mở.',401:'Đăng nhập/session DRES không hợp lệ hoặc hết hạn.',
                403:'Tài khoản chưa có quyền tham gia lượt thi.',404:'Không tìm thấy endpoint/lượt thi DRES.',
                412:'DRES từ chối bài gửi (ví dụ trùng bài).'}
            raise HTTPException(response.status_code if response.status_code in descriptions else 502,
                descriptions.get(response.status_code, 'DRES không trả kết quả hợp lệ; kiểm tra server URL/API v2.'))
        return response.status_code, response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, 'Không kết nối được DRES hoặc phản hồi không phải JSON. Kiểm tra URL/mạng.') from exc


def install_dres(app, catalog):
    sessions = {}
    lock = threading.Lock()

    def state(request):
        key = request.cookies.get(COOKIE, '')
        with lock:
            session = sessions.get(key)
            if not session or session['expires'] < time.monotonic():
                sessions.pop(key, None)
                raise HTTPException(401, 'Chưa đăng nhập DRES hoặc phiên local đã hết hạn.')
        return session

    def public(session):
        return {'connected': True, 'server': session['server'], 'username': session['username'],
                'evaluation_id': session['evaluation_id']}

    def evaluations(session):
        _, values = remote(session['server'], 'GET', '/api/v2/client/evaluation/list', session['token'])
        if not isinstance(values, list): raise HTTPException(502, 'Danh sách lượt thi DRES không hợp lệ.')
        return [{'id': v['id'], 'name': v.get('name',v['id']), 'status': v.get('status','')} for v in values]

    @app.middleware('http')
    async def same_origin(request, call_next):
        if request.url.path.startswith('/api/images/dres') and request.method != 'GET':
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return Response('Cross-origin DRES request rejected',status_code=403)
        return await call_next(request)

    @app.post('/api/images/dres/login')
    def login(body: Login, response: Response, request: Request):
        server = server_url(body.server)
        token = body.session_id.get_secret_value().strip()
        if token:
            _, user = remote(server,'GET','/api/v2/user',token)
        else:
            if not body.username.strip() or not body.password.get_secret_value():
                raise HTTPException(422,'Nhập user/password hoặc Session ID.')
            _, user = remote(server,'POST','/api/v2/login',payload={
                'username':body.username.strip(),'password':body.password.get_secret_value()})
            token = user.get('sessionId','')
        if not token or not user.get('username'):
            raise HTTPException(401,'DRES chưa xác nhận tài khoản/session đăng nhập.')
        session = {'server':server,'token':token,'username':user['username'],
                   'evaluation_id':'','expires':time.monotonic()+TTL}
        runs = evaluations(session)
        key = secrets.token_urlsafe(32)
        with lock:
            for old in list(sessions):
                if sessions[old]['expires'] < time.monotonic(): sessions.pop(old)
            if len(sessions) >= 100: raise HTTPException(503,'Quá nhiều phiên DRES local.')
            sessions.pop(request.cookies.get(COOKIE,''),None)
            sessions[key]=session
        response.set_cookie(COOKIE,key,httponly=True,samesite='strict',secure=request.url.scheme=='https',max_age=TTL)
        return {**public(session),'evaluations':runs}

    @app.get('/api/images/dres/session')
    def session_status(request: Request):
        try: return public(state(request))
        except HTTPException: return {'connected':False}

    @app.get('/api/images/dres/evaluations')
    def list_evaluations(request: Request):
        return evaluations(state(request))

    @app.post('/api/images/dres/evaluation')
    def select(body: Selection, request: Request):
        session = state(request)
        if body.evaluation_id not in {r['id'] for r in evaluations(session)}:
            raise HTTPException(422,'Lượt thi không có trong danh sách tài khoản DRES.')
        session['evaluation_id']=body.evaluation_id
        return public(session)

    @app.post('/api/images/dres/logout')
    def logout(request: Request, response: Response):
        with lock: sessions.pop(request.cookies.get(COOKIE,''),None)
        response.delete_cookie(COOKIE)
        return {'connected':False}

    @app.post('/api/images/dres/submit')
    def submit(body: Submission, request: Request):
        session=state(request)
        if not session['evaluation_id']: raise HTTPException(422,'Chọn Evaluation ID trước khi gửi.')
        if body.image_id not in catalog().index: raise HTTPException(404,'Ảnh không tồn tại.')
        answer={'mediaItemName':body.media_item_name.strip()}
        if not answer['mediaItemName']: raise HTTPException(422,'Tên ảnh DRES không được trống.')
        if body.collection.strip(): answer['mediaItemCollectionName']=body.collection.strip()
        answer_set={'answers':[answer]}
        if body.task_id.strip(): answer_set['taskId']=body.task_id.strip()
        status,result=remote(session['server'],'POST','/api/v2/submit/'+quote(session['evaluation_id'],safe=''),
            session['token'],{'answerSets':[answer_set]})
        # DRES acceptance is not equivalent to a CORRECT verdict.
        return {'http_status':status,'status':result.get('status'),
                'submission':result.get('submission'),'description':str(result.get('description',''))[:1000]}
