"""Fast query editing via the authenticated Codex CLI, with structured output."""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

from pydantic import BaseModel, Field

MODEL = 'gpt-6-luna'


class AgentUnavailable(RuntimeError): pass


class AgentInput(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)
    current: dict = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list, max_length=6)


SYSTEM = """Edit image-search CURRENT using instruction. Return only changed fields in edits; empty edits if unclear.
CURRENT is authoritative: preserve unrelated filters. New scene replaces text/conflicting objects; follow-ups edit only requested conditions. Semantic text concise English; filter-only requests need no semantic text.
Allowed labels/colors/days/sources are in schema. Never invent. Bare N people means exactly N (min=max); at least N min=N,max=null; at most N min=0,max=N; more/less than are strict. All object constraints AND; default count_source=detected. Metadata only when explicit, cannot count colors, missing counts unknown. Red shirt means shirt/red, not person/red. Preserve existing color thresholds (host handles).
Hours inclusive 0..23. Frame is not seconds. Event order/identity/spatial relations stay semantic with short Vietnamese warning, never enforced filters. Unsupported labels stay semantic with warning. Ambiguous: ask short Vietnamese question in clarification, edits=[]; otherwise clarification="". No tools/files. No prose beyond JSON."""


def output_schema(context):
    nullable = lambda values: {'anyOf':[{'type':'string','enum':values},{'type':'null'}]}
    obj = {'type':'object','additionalProperties':False,'properties':{
        'label':{'type':'string','enum':context['labels']},
        'color':nullable(context['colors']),
        'min_count':{'type':'integer','minimum':0,'maximum':100},
        'max_count':{'anyOf':[{'type':'integer','minimum':0,'maximum':100},{'type':'null'}]},
        'count_source':{'type':'string','enum':['detected','metadata']},
    },'required':['label','color','min_count','max_count','count_source']}
    edit = {'type':'object','additionalProperties':False,'properties':{
        'field':{'type':'string','enum':['text','day','source','hour_from','hour_to','objects']},
        'value':{'anyOf':[{'type':'string','maxLength':600},{'type':'integer','minimum':0,'maximum':23},
            {'type':'null'},{'type':'array','items':obj,'maxItems':12}]},
    },'required':['field','value']}
    return {'type':'object','additionalProperties':False,'properties':{
        'edits':{'type':'array','items':edit,'maxItems':6},
        'clarification':{'type':'string','maxLength':200},
        'warnings':{'type':'array','items':{'type':'string','maxLength':200},'maxItems':3},
    },'required':['edits','clarification','warnings']}


def local_edit(instruction, current, context):
    """Only consume complete, unambiguous filter commands; never partial-match scenes."""
    text = ''.join(c for c in unicodedata.normalize('NFD',instruction.lower()) if unicodedata.category(c) != 'Mn').replace('đ','d')
    text = text.strip().rstrip('.!').strip()
    query = current.model_dump()
    if re.fullmatch(r'(?:bo|xoa) (?:loc )?mau',text):
        for obj in query['objects']: obj['color'] = None
        return query, 'Đã bỏ ràng buộc màu.'
    match = re.fullmatch(r'(?:(?:them|doi thanh|chi|tim|loc) )?(?:(dung|chinh xac|it nhat|toi thieu|toi da|nhieu nhat|hon|duoi|it hon) )?(\d{1,2}) (nguoi|person|people)',text)
    if not match or 'person' not in context['labels']: return None
    mode,n = match[1],int(match[2])
    minimum,maximum = n,n
    if mode in ('it nhat','toi thieu'): maximum=None
    elif mode in ('toi da','nhieu nhat'): minimum=0
    elif mode=='hon': minimum,maximum=n+1,None
    elif mode in ('duoi','it hon'):
        if n==0: return None
        minimum,maximum=0,n-1
    obj={'label':'person','color':None,'min_count':minimum,'max_count':maximum,'count_source':'detected','min_color_fraction':.15}
    # Preserve color-specific person filters and all unrelated constraints.
    query['objects']=[f for f in query['objects'] if not(f['label']=='person' and f['color'] is None)]+[obj]
    query['offset']=0
    return query, 'Đã cập nhật số người (detection).'


class QueryAgent:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.executable = os.environ.get('CODEX_QUERY_EXECUTABLE') or shutil.which('codex')

    def parse(self, request, context):
        from search_api import SearchRequest
        started = time.monotonic()
        current = SearchRequest.model_validate(request.current)
        local = local_edit(request.instruction,current,context)
        if local:
            query,message=local
            query=SearchRequest.model_validate(query)
            return {'model':MODEL,'engine':'local','message':message,'warnings':[],
                'needs_clarification':False,'query':query.model_dump(),'elapsedMs':round((time.monotonic()-started)*1000),
                'usage':{'input_tokens':0,'output_tokens':0}}
        key=json.dumps([request.instruction,current.model_dump(),context],sort_keys=True,ensure_ascii=False)
        cached=self.cache.get(key)
        if cached:
            reply=deepcopy(cached)
            reply.update(engine='cache',elapsedMs=round((time.monotonic()-started)*1000),usage={'input_tokens':0,'output_tokens':0})
            return reply
        if not self.executable:
            raise AgentUnavailable('Không tìm thấy Codex CLI. Cài/đăng nhập Codex để dùng parser Luna.')
        fields = ['text','day','source','hour_from','hour_to','objects']
        clean_current = {k:v for k,v in current.model_dump().items() if k in fields and v not in (None,'',[])}
        for obj in clean_current.get('objects',[]): obj.pop('min_color_fraction',None)
        # Current structured state replaces conversation history; avoid replaying prose.
        prompt = json.dumps({'instruction':request.instruction,'current':clean_current,'days':context['days'],'sources':context['sources']},ensure_ascii=False,separators=(',',':'))
        if not self.lock.acquire(blocking=False):
            raise AgentUnavailable('Agent đang xử lý một yêu cầu. Thử lại sau vài giây.')
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix='image-query-agent-') as temporary:
                root = Path(temporary)
                schema = root/'schema.json'
                output = root/'response.json'
                schema.write_text(json.dumps(output_schema(context),separators=(',',':')),encoding='utf-8')
                instructions=root/'instructions.txt'
                instructions.write_text(SYSTEM,encoding='utf-8')
                command = [self.executable,'exec','--ignore-user-config','--ephemeral',
                    '--skip-git-repo-check','--sandbox','read-only','--model',MODEL,
                    '-c','model_reasoning_effort="low"','-c','approval_policy="never"',
                    '-c','project_doc_max_bytes=0','-c','model_instructions_file='+json.dumps(str(instructions)),
                    '-c','model_verbosity="low"','-c','model_reasoning_summary="none"',
                    '-c','skills.max_context_tokens=1','--enable','skip_host_skill_discovery',
                    '--disable','shell_tool','--disable','apps',
                    '--disable','browser_use','--disable','computer_use','--disable','view_image',
                    '--color','never','--json','--output-schema',str(schema),'-o',str(output),'-']
                for feature in ('multi_agent','plugins','skill_search','code_mode_host','unified_exec',
                                'sleep_tool','image_generation','goals','workspace_dependencies'):
                    command.extend(['--disable',feature])
                # No Kaggle token, GPU key, API keys or project secrets are inherited.
                allowed_env = {'SYSTEMROOT','SYSTEMDRIVE','COMSPEC','PATH','PATHEXT','TEMP','TMP',
                               'USERPROFILE','APPDATA','LOCALAPPDATA','HOMEDRIVE','HOMEPATH','CODEX_HOME',
                               'HTTP_PROXY','HTTPS_PROXY','NO_PROXY','SSL_CERT_FILE'}
                env = {k:v for k,v in os.environ.items() if k.upper() in allowed_env}
                result = subprocess.run(command,input=prompt,capture_output=True,text=True,encoding='utf-8',
                    errors='replace',cwd=root,env=env,timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if result.returncode or not output.exists():
                    # Keep diagnostic details local; never send CLI internals/auth to the browser.
                    diagnostic = Path(__file__).parent/'.work'/'query_agent_error.log'
                    diagnostic.parent.mkdir(exist_ok=True)
                    diagnostic.write_text(result.stderr[-10000:]+'\n'+result.stdout[-10000:],encoding='utf-8')
                    raise AgentUnavailable('Codex GPT-6 Luna chưa trả kết quả. Kiểm tra đăng nhập/model và thử lại.')
                answer = json.loads(output.read_text(encoding='utf-8'))
                edits={}
                for edit in answer['edits']:
                    field=edit['field']
                    if field not in fields or field in edits: raise ValueError('Invalid/duplicate edit')
                    value=edit['value']
                    if field=='text' and not isinstance(value,str): raise ValueError('Invalid text')
                    if field in ('day','source') and value is not None and not isinstance(value,str): raise ValueError('Invalid facet')
                    if field in ('hour_from','hour_to') and value is not None and type(value) is not int: raise ValueError('Invalid hour')
                    if field=='objects' and not isinstance(value,list): raise ValueError('Invalid objects')
                    edits[field]=value
                for proposed in edits.get('objects',[]):
                    previous = next((f for f in current.objects if f.label == proposed['label'] and f.color == proposed.get('color') and f.count_source == proposed['count_source']), None)
                    proposed['min_color_fraction'] = previous.min_color_fraction if previous else .15
                query = SearchRequest.model_validate({**current.model_dump(), **edits, 'offset':0})
                for f in query.objects:
                    if f.label not in context['labels'] or f.color and f.color not in context['colors']:
                        raise ValueError('Unknown object/color')
                if query.day and query.day not in context['days'] or query.source and query.source not in context['sources']:
                    raise ValueError('Unknown day/source')
                usage={}
                for line in result.stdout.splitlines():
                    try: event=json.loads(line)
                    except ValueError: continue
                    if event.get('type')=='turn.completed': usage=event.get('usage',{})
                clarification=str(answer['clarification'])[:200]
                names={'text':'nội dung','day':'ngày','source':'nguồn','hour_from':'giờ bắt đầu','hour_to':'giờ kết thúc','objects':'object/màu/số lượng'}
                message=clarification or ('Đã chỉnh '+', '.join(names[k] for k in edits)+'.' if edits else 'Giữ nguyên điều kiện.')
                reply={'model':MODEL,'engine':'luna','message':message,
                       'warnings':[str(x)[:200] for x in answer['warnings'][:3]],
                       'needs_clarification':bool(clarification),
                       'query':current.model_dump() if clarification else query.model_dump(),
                       'elapsedMs':round((time.monotonic()-started)*1000),'usage':usage}
                self.cache[key]=deepcopy(reply)
                if len(self.cache)>128: self.cache.popitem(last=False)
                return reply
        except subprocess.TimeoutExpired as exc:
            raise AgentUnavailable('Luna phản hồi quá thời gian. Điều kiện hiện tại được giữ nguyên; có thể thử lại.') from exc
        except (ValueError,KeyError,TypeError) as exc:
            raise AgentUnavailable('Agent trả điều kiện không hợp lệ. Điều kiện hiện tại được giữ nguyên.') from exc
        finally:
            self.lock.release()
