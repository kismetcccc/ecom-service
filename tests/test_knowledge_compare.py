"""Metric edge cases: alternate evidence, missing scores, and paired grouping."""
import json
import asyncio
from types import SimpleNamespace

import pytest

from app.evaluation.knowledge_compare import evidence_metrics, valid_grade, LivePool
from app.evaluation.knowledge_report import summarize, paired_bootstrap


def test_evidence_alternatives_and_empty_gold():
    case={'required_evidence_groups':[['a','b'],['c']], 'relevant_chunk_ids':['a','b','c']}
    result=evidence_metrics(case,{'results':[{'chunk_id':x} for x in ['b','b','c']]})
    assert result['evidence_recall']==1
    assert result['complete_evidence']==1
    assert result['returned_chunks']==2
    assert result['recall_at_k']==2/3  # Binary top-k metric is distinct from group coverage.
    empty=evidence_metrics({'required_evidence_groups':[], 'relevant_chunk_ids':[]},{})
    assert empty['evidence_recall'] is None and empty['complete_evidence'] is None


def test_judge_rejects_boolean_score_and_out_of_range_points():
    grade=dict(correctness=2, covered_points=[0], false_answer=False, false_completion=False,
               unsupported_claim=False, citation_supported=None, reason='符合')
    assert valid_grade(grade,1)
    assert not valid_grade({**grade,'correctness':True},1)
    assert not valid_grade({**grade,'covered_points':[1]},1)


def test_missing_answer_and_judge_do_not_become_zero_or_success(tmp_path,monkeypatch):
    case=dict(id='X', family_id='X',split_group='topic',split='test',category='unknown',
              answerability='unanswerable',query='未知',answer_points=['应说明未知'],relevant_chunk_ids=[])
    monkeypatch.setattr('app.evaluation.knowledge_report.cases',lambda:[case])
    row=dict(case_id='X',mode='rag',provider='rag',fallback_reason=None,retrieval_ms=10,
             context_chars=0,evidence_recall=None,complete_evidence=None,recall_at_k=None,
             mrr_at_k=None,ndcg_at_k=None,returned_chunks=0,answer_key='answer',result={'success':True})
    (tmp_path/'metadata.json').write_text('{}',encoding='utf-8')
    (tmp_path/'retrieval.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    summary,rows,_=summarize(tmp_path)
    metric=summary['mode_metrics']['rag']
    assert metric['answers_n']==0 and metric['judged_n']==0
    assert metric['correct_rate'] is None
    assert metric['no_answer_false_answer_rate'] is None
    assert rows[0]['correct'] is None
    assert summary['judge_calibration']['reject_precision'] is None


def test_cluster_bootstrap_pairs_only_common_scored_cases():
    rows=[dict(case_id='a',mode='rag',split_group='g1',correct=0),
          dict(case_id='a',mode='wiki',split_group='g1',correct=1),
          dict(case_id='b',mode='rag',split_group='g2',correct=1),
          dict(case_id='b',mode='wiki',split_group='g2',correct=None)]
    result=paired_bootstrap(rows,'rag','wiki','correct')
    assert result['cluster_count']==1
    assert result['cluster_macro_difference']==1
    assert result['bootstrap95']==[1,1]


def test_provider_denial_stops_instead_of_continuing_batch(tmp_path):
    class Denied(Exception):
        status_code=403
        body={'error':{'code':'AllocationQuota.FreeTierOnly'}}
    async def denied(**kwargs): raise Denied('not recorded')
    async def pace(): pass
    async def run():
        pool=object.__new__(LivePool)
        pool.output=tmp_path
        pool.semaphore=asyncio.Semaphore(1)
        pool.pace=pace
        pool.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=denied)))
        await pool.call('answer','example','prompt',{},10)
    with pytest.raises(RuntimeError,match='AllocationQuota.FreeTierOnly'):
        asyncio.run(run())
    rows=(tmp_path/'api_calls.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(rows)==1
    assert json.loads(rows[0])['http_status']==403
