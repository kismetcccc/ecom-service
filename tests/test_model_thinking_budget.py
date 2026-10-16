import json

import httpx
import pytest

from app.config.llm_transport import GuardedModelClient, ModelGate
from app.config.settings import settings


@pytest.mark.parametrize('explicit,expected', [(None,128),(1024,128),(64,64)])
def test_required_thinking_preserves_tools_and_request_metadata(monkeypatch, explicit, expected):
    monkeypatch.setattr(settings,'llm_required_thinking_budget',128)
    payload={'model':'qwen3.8-2.4t-a95b','messages':[{'role':'user','content':'查询订单'}],
             'tools':[{'type':'function','function':{'name':'query_order','parameters':{'type':'object'}}}],
             'max_tokens':4000,'enable_thinking':False}
    if explicit is not None: payload['thinking_budget']=explicit
    def handler(request):
        body=json.loads(request.content)
        assert body['thinking_budget']==expected
        assert body['enable_thinking'] is True
        assert body['tools']==payload['tools']
        assert body['messages']==payload['messages']
        assert body['max_tokens']==4000
        assert int(request.headers['content-length'])==len(request.content)
        assert request.headers['authorization']=='Bearer fake-test-key'
        assert request.extensions['timeout']['read']==17
        return httpx.Response(200,json={'ok':True})
    with GuardedModelClient(transport=httpx.MockTransport(handler),gate=ModelGate()) as client:
        response=client.post('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                             json=payload,headers={'authorization':'Bearer fake-test-key'},timeout=17)
        assert response.json()=={'ok':True}


@pytest.mark.parametrize('host,model', [('api.openai.com','qwen3.8-2.4t-a95b'),
                                     ('dashscope.aliyuncs.com','another-model')])
def test_unrelated_providers_and_models_unchanged(host,model):
    payload={'model':model,'messages':[]}
    def handler(request):
        assert json.loads(request.content)==payload
        return httpx.Response(200,json={'ok':True})
    with GuardedModelClient(transport=httpx.MockTransport(handler),gate=ModelGate()) as client:
        client.post(f'https://{host}/v1/chat/completions',json=payload)


@pytest.mark.parametrize('parse', [False, True])
def test_flash_disables_thinking_for_sdk_create_and_parse(parse):
    from openai import OpenAI
    from pydantic import BaseModel

    class Verdict(BaseModel):
        approved: bool

    def handler(request):
        body = json.loads(request.content)
        assert body['enable_thinking'] is False
        assert 'thinking_budget' not in body
        assert body['response_format']['type'] == ('json_schema' if parse else 'json_object')
        assert body['messages'][0]['content'] == 'Return JSON verdict'
        assert request.extensions['timeout']['read'] == 17
        assert int(request.headers['content-length']) == len(request.content)
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'qwen3.8-flash', 'choices': [{'index': 0,
            'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{"approved":true}'}}]})

    with OpenAI(api_key='fake', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                http_client=GuardedModelClient(transport=httpx.MockTransport(handler), gate=ModelGate()),
                max_retries=0) as client:
        method = client.beta.chat.completions.parse if parse else client.chat.completions.create
        result = method(model='qwen3.8-flash', messages=[{'role': 'user', 'content': 'Return JSON verdict'}],
                        response_format=Verdict if parse else {'type': 'json_object'}, timeout=17,
                        extra_body={'enable_thinking': True, 'thinking_budget': 8000})
        assert json.loads(result.choices[0].message.content)['approved'] is True
