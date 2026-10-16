"""Execute L01-L32 in isolated directories; no live indexes or model calls."""
from dataclasses import replace
from datetime import date
import json

import pytest

from app.agent import wiki_runtime as wiki
from app.agent.tools import knowledge
from app.agent.rag import policy
from app.agent.rag.chunker import Chunk
from app.config.settings import settings
from app.evaluation.knowledge_dataset import DATA, read_jsonl
from app.multi_agent.wiki import Citation, WikiDraft, WikiPage, WikiSnapshot, source_hash, analyze_impact


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    kb=tmp_path/'kb'; kb.mkdir()
    (kb/'policy.md').write_text('# Policy\n\n退货需要订单号。',encoding='utf-8')
    (kb/'policies.json').write_text(json.dumps({'policy':{'status':'demo','version':'demo-1'}}),encoding='utf-8')
    monkeypatch.setattr(settings,'kb_dir',str(kb))
    monkeypatch.setattr(settings,'wiki_store_dir',str(tmp_path/'wiki'))
    monkeypatch.setattr(settings,'knowledge_mode','rag')
    monkeypatch.setattr(settings,'rag_allow_demo_policies',True)
    sources=wiki.current_sources()
    c=next(iter(sources.values()))
    draft=WikiSnapshot(created_at='2026-10-15T00:00:00Z',source_hashes={k:source_hash(v) for k,v in sources.items()},
        content=WikiDraft(pages=[WikiPage(title='退货流程',summary='需核对订单',steps=['核对订单'],
            citations=[Citation(chunk_id=c.chunk_id,quote='退货需要订单号。')])]))
    calls=[]
    monkeypatch.setattr(knowledge,'_search_knowledge_impl',lambda *args: calls.append(args) or {'success':True,'results':[]})
    return kb,draft,calls


SPECS=read_jsonl(DATA/'lifecycle_cases.jsonl')


@pytest.mark.parametrize('spec',SPECS,ids=[r['id'] for r in SPECS])
def test_lifecycle_spec(spec,isolated,monkeypatch):
    ident=spec['id']; kb,draft,calls=isolated
    catalog=kb/'policies.json'
    if ident in ('L06','L07','L08'):
        catalog.write_text(json.dumps({'policy':{'status':'published','version':'1','policy_id':'p',
            'owner':'test','source_url':'https://example.test/p','effective_from':'2020-01-01','effective_to':'2099-01-01'}}),encoding='utf-8')
        draft.source_hashes={k:source_hash(v) for k,v in wiki.current_sources().items()}
    if ident in ('L21','L22'):
        monkeypatch.setattr(settings,'knowledge_mode','wiki' if ident=='L21' else 'wiki_first')
        result=knowledge.search_knowledge('退货')
        if ident=='L21':
            assert result['error_code']=='INDEX_MISSING' and not calls
        else:
            assert result['provider']=='rag' and result['wiki_fallback_reason']=='INDEX_MISSING' and len(calls)==1
        return
    if ident=='L28':
        monkeypatch.setattr(settings,'wiki_store_dir',str(kb/'wiki'))
        with pytest.raises(ValueError): wiki.store_dir()
        assert not (kb/'wiki').exists()
        return
    if ident in ('L29','L30','L31','L32'):
        count=24 if ident=='L32' else 3
        sources={f'c{i}':Chunk(f'c{i}','test',str(i),f'共同查询证据{i}。') for i in range(count)}
        pages=[]
        for i in range(3):
            indexes=range(i*8,(i+1)*8) if ident=='L32' else [i]
            pages.append(WikiPage(title=chr(65+i),summary='共同查询',steps=['核对'],
                citations=[Citation(chunk_id=f'c{j}',quote=sources[f'c{j}'].text) for j in indexes]))
        pages[1].related_titles=['A']; pages[2].related_titles=['B']
        if ident=='L30': pages[0].related_titles=['B']
        snap=WikiSnapshot(created_at='2026-10-15',source_hashes={k:source_hash(v) for k,v in sources.items()},content=WikiDraft(pages=pages))
        if ident=='L32':
            monkeypatch.setattr(wiki,'current_sources',lambda:sources)
            release=wiki.publish(snap,'test')
            result=wiki.search_wiki('共同查询',3,wiki.KnowledgeState(mode='wiki',active_release=release.release_id))
            assert result['success'] and len(result['results'])==16 and len(result['wiki_pages'])==2
            ids={r['chunk_id'] for r in result['results']}
            assert all(set(p['source_chunk_ids'])<=ids for p in result['wiki_pages'])
        else:
            changed={**sources,'new':Chunk('new','test','new','新增')} if ident=='L31' else {**sources,'c0':replace(sources['c0'],version='changed')}
            report=analyze_impact(snap,changed)
            if ident=='L31': assert report['new_source_review_required'] and report['corpus_review_required']
            else:
                assert [p['status'] for p in report['pages']]==['direct','dependent','dependent']
                assert report['pages'][2]['dependency_path']==['C','B','A']
        return
    release=wiki.publish(draft,'test')
    control=wiki.store_dir()/'control.json'
    before=control.read_bytes()
    if ident in {f'L{i:02d}' for i in range(1,9)}:
        wiki.switch_mode('wiki')
        if ident=='L01': (kb/'policy.md').write_text('# Policy\n\n新规则。',encoding='utf-8')
        elif ident=='L02':
            obj=json.loads(catalog.read_text(encoding='utf-8')); obj['policy']['version']='changed'; catalog.write_text(json.dumps(obj),encoding='utf-8')
        elif ident=='L03': (kb/'policy.md').unlink()
        elif ident=='L04':
            (kb/'new.md').write_text('# New\n\n新增政策。',encoding='utf-8')
            obj=json.loads(catalog.read_text(encoding='utf-8')); obj['new']={'status':'demo','version':'demo-1'}; catalog.write_text(json.dumps(obj),encoding='utf-8')
        elif ident=='L05': monkeypatch.setattr(settings,'rag_allow_demo_policies',False)
        elif ident=='L06':
            class FutureDate(date):
                @classmethod
                def today(cls): return cls(2099,1,2)
            monkeypatch.setattr(policy,'date',FutureDate)
        else:
            obj=json.loads(catalog.read_text(encoding='utf-8'))
            obj['policy']['status' if ident=='L07' else 'owner']='retired' if ident=='L07' else 'new-owner'
            catalog.write_text(json.dumps(obj),encoding='utf-8')
        assert knowledge.search_knowledge('退货')['error_code']=='UNAVAILABLE'
        assert not calls
    elif ident in {f'L{i:02d}' for i in range(9,16)}:
        if ident=='L09': draft.content.pages[0].open_questions=['未决']
        if ident=='L11': draft.content.pages[0].citations[0].quote='凭空编造'
        if ident=='L12': draft.content.pages[0].citations[0].chunk_id='missing'
        if ident=='L13': draft.content.pages.append(draft.content.pages[0].model_copy(deep=True))
        if ident=='L14': draft.content.pages[0].related_titles=[draft.content.pages[0].title]
        if ident=='L15': draft.content.pages[0].related_titles=['missing']
        with pytest.raises(ValueError): wiki.publish(draft,'   ' if ident=='L10' else 'test',activate_release=False)
        assert control.read_bytes()==before
    elif ident in ('L16','L17','L18'):
        candidate=wiki.publish(draft,'test',activate_release=ident=='L17')
        if ident=='L17':
            assert wiki.read_state().active_release==candidate.release_id and wiki.read_state().mode=='rag'
        else:
            result=wiki.search_wiki('退货',3,wiki.KnowledgeState(mode='wiki',active_release=candidate.release_id))
            assert result['success'] and result['index_version']==candidate.release_id
            assert control.read_bytes()==before and wiki.read_state().active_release==release.release_id
    elif ident=='L19':
        (kb/'policy.md').write_text('来源改变',encoding='utf-8')
        with pytest.raises(ValueError): wiki.activate(release.release_id)
        assert control.read_bytes()==before
    elif ident=='L20':
        def fail(*args): raise OSError('injected atomic replace failure')
        monkeypatch.setattr(wiki.os,'replace',fail)
        with pytest.raises(OSError): wiki.publish(draft,'test')
        assert control.read_bytes()==before
    elif ident in ('L23','L24'):
        wiki.switch_mode('wiki_first')
        if ident=='L23': (kb/'policy.md').write_text('changed',encoding='utf-8')
        result=knowledge.search_knowledge('退货' if ident=='L23' else 'zzzzzz')
        assert result['provider']=='rag' and len(calls)==1
        assert result['wiki_fallback_reason']==('UNAVAILABLE' if ident=='L23' else 'NO_MATCH')
    elif ident in ('L25','L26'):
        control.write_text('{broken',encoding='utf-8')
        if ident=='L25': assert knowledge.search_knowledge('退货')['error_code']=='UNAVAILABLE' and not calls
        else:
            wiki.switch_mode('rag')
            assert wiki.read_state().mode=='rag' and wiki.load_release(release.release_id)
    elif ident=='L27':
        with pytest.raises(ValueError): wiki.load_release('../../outside')
    else: raise AssertionError('Unimplemented dataset scenario '+ident)
