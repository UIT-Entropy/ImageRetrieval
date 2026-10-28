import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from query_agent import AgentInput, QueryAgent, AgentUnavailable, MODEL, output_schema


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.context={'labels':['person','shirt','cup'],'colors':['red','blue'],'days':['day1'],'sources':['Kitchen']}
        self.current={'text':'A person washing a cup.','reference_id':'a','objects':[],'hour_from':10}
        self.answer={'warnings':[],'clarification':'','edits':[{'field':'objects','value':[
            {'label':'person','color':None,'min_count':2,'max_count':2,'count_source':'detected'}]}]}
        self.agent=QueryAgent()
        self.agent.executable='codex.exe'

    def runner(self,command,**kwargs):
        self.assertEqual(command[command.index('--model')+1],MODEL)
        self.assertEqual(command[command.index('--sandbox')+1],'read-only')
        self.assertIn('--ignore-user-config',command)
        self.assertIn('--ephemeral',command)
        self.assertNotIn('KAGGLE_API_TOKEN',kwargs['env'])
        self.assertNotIn('IMAGE_ENCODER_KEY',kwargs['env'])
        Path(command[command.index('-o')+1]).write_text(json.dumps(self.answer),encoding='utf-8')
        return SimpleNamespace(returncode=0,stdout='',stderr='')

    def test_structured_edit_preserves_image_reference_and_uses_requested_model(self):
        with patch.dict(os.environ,{'KAGGLE_API_TOKEN':'do-not-inherit','IMAGE_ENCODER_KEY':'do-not-inherit'}),patch('query_agent.subprocess.run',side_effect=self.runner):
            reply=self.agent.parse(AgentInput(instruction='Người bên cạnh đang rửa cốc, thêm đúng 2 người',current=self.current),self.context)
        self.assertEqual(reply['model'],'gpt-6-luna')
        self.assertEqual(reply['query']['reference_id'],'a')
        self.assertEqual(reply['query']['objects'][0]['max_count'],2)
        self.assertEqual(reply['query']['hour_from'],10)
        self.assertEqual(reply['query']['objects'][0]['min_color_fraction'],.15)

    def test_clarification_keeps_current_query(self):
        self.answer['clarification']='Lúc nào?'
        with patch('query_agent.subprocess.run',side_effect=self.runner):
            reply=self.agent.parse(AgentInput(instruction='Tìm lúc đó',current=self.current),self.context)
        self.assertEqual(reply['query']['objects'],[])
        self.assertEqual(reply['query']['text'],self.current['text'])

    def test_invalid_agent_counts_and_timeout_fail_without_fallback(self):
        self.answer['edits'][0]['value'][0]['max_count']=1
        with patch('query_agent.subprocess.run',side_effect=self.runner):
            with self.assertRaises(AgentUnavailable): self.agent.parse(AgentInput(instruction='Người bên cạnh đang rửa cốc',current=self.current),self.context)
        with patch('query_agent.subprocess.run',side_effect=subprocess.TimeoutExpired('codex',60)):
            with self.assertRaisesRegex(AgentUnavailable,'quá thời gian'): self.agent.parse(AgentInput(instruction='Người bên cạnh đang rửa cốc',current=self.current),self.context)

    def test_schema_only_exposes_allowed_labels_and_query_fields(self):
        schema=output_schema(self.context)
        edit=schema['properties']['edits']['items']
        self.assertFalse(edit['additionalProperties'])
        self.assertNotIn('reference_id',edit['properties']['field']['enum'])
        self.assertEqual(edit['properties']['value']['anyOf'][3]['items']['properties']['label']['enum'],self.context['labels'])

    def test_local_edits_never_call_codex_and_preserve_other_conditions(self):
        for instruction,minimum,maximum in [('đúng 2 người',2,2),('ít nhất 3 người',3,None),('dưới 2 người',0,1)]:
            with patch('query_agent.subprocess.run') as run:
                reply=self.agent.parse(AgentInput(instruction=instruction,current=self.current),self.context)
            run.assert_not_called()
            self.assertEqual(reply['engine'],'local')
            self.assertEqual(reply['usage']['input_tokens'],0)
            self.assertEqual(reply['query']['text'],self.current['text'])
            self.assertEqual(reply['query']['hour_from'],10)
            obj=reply['query']['objects'][0]
            self.assertEqual((obj['min_count'],obj['max_count']),(minimum,maximum))

    def test_cache_is_scoped_to_current_state(self):
        request=AgentInput(instruction='Người đang rửa cốc, thêm 2 người',current=self.current)
        with patch('query_agent.subprocess.run',side_effect=self.runner) as run:
            self.agent.parse(request,self.context)
            reply=self.agent.parse(request,self.context)
            self.assertEqual(reply['engine'],'cache')
            self.assertEqual(run.call_count,1)
            request.current={**self.current,'hour_from':11}
            self.agent.parse(request,self.context)
            self.assertEqual(run.call_count,2)

    def test_remove_color_preserves_count_source_and_threshold(self):
        current={**self.current,'objects':[{'label':'shirt','color':'red','min_count':2,'max_count':2,'min_color_fraction':.3}]}
        with patch('query_agent.subprocess.run') as run:
            reply=self.agent.parse(AgentInput(instruction='bỏ màu',current=current),self.context)
        run.assert_not_called()
        obj=reply['query']['objects'][0]
        self.assertIsNone(obj['color'])
        self.assertEqual(obj['max_count'],2)
        self.assertEqual(obj['min_color_fraction'],.3)



if __name__=='__main__': unittest.main()
