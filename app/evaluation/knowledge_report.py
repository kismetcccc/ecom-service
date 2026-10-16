"""Summarize recorded API results only; missing judgements remain unknown."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import xml.etree.ElementTree as ET

from app.evaluation.knowledge_compare import cases, load_rows, MODES
from app.evaluation.knowledge_dataset import dump, DATA, read_jsonl
from app.evaluation.statistics import percentile


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def format_value(value, percent=False):
    if value is None: return '未测 / 不适用'
    return f'{value*100:.2f}%' if percent else f'{value:.2f}'


def family_mean(rows, field):
    groups=defaultdict(list)
    for row in rows:
        if row.get(field) is not None: groups[row['family_id']].append(row[field])
    return mean([mean(v) for v in groups.values()])


def paired_bootstrap(rows, first, second, field):
    paired=defaultdict(dict)
    for row in rows:
        if row['mode'] in (first,second) and row.get(field) is not None:
            paired[row['case_id']][row['mode']]=row
    groups=defaultdict(list)
    for pair in paired.values():
        if len(pair)==2:
            groups[pair[first]['split_group']].append(pair[second][field]-pair[first][field])
    values=[mean(v) for v in groups.values()]
    if not values: return None
    rng=random.Random(20261015)
    boot=[mean(rng.choices(values,k=len(values))) for _ in range(2000)]
    return {'direction':second+' minus '+first,'cluster_unit':'split_group',
            'cluster_count':len(values),'cluster_macro_difference':mean(values),
            'bootstrap95':[percentile(boot,2.5),percentile(boot,97.5)]}


def summarize(output):
    case_map={r['id']:r for r in cases()}
    retrieval=load_rows(output/'retrieval.jsonl')
    answers={r['key']:r for r in load_rows(output/'answers.jsonl')}
    judgments={r['case_id']:r for r in load_rows(output/'judgements.jsonl')}
    meta=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    rows=[]
    for r in retrieval:
        case=case_map[r['case_id']]
        a=answers.get(r['answer_key'],{})
        j=judgments.get(r['case_id'],{})
        grade=None
        if j and not j.get('error'):
            grade=next((g for g in j['data']['grades'] if j['label_to_answer_key'][g['label']]==r['answer_key']),None)
        row={**{k:r[k] for k in ('case_id','mode','provider','fallback_reason','retrieval_ms','context_chars',
                                'evidence_recall','complete_evidence','recall_at_k','mrr_at_k','ndcg_at_k','returned_chunks','answer_key')},
             **{k:case[k] for k in ('family_id','split_group','split','category','answerability')},
             'style':case.get('style','unknown'),
             'answer_model':a.get('request_model',meta.get('legacy_answer_model',meta.get('answer_model'))) if a else None,
             'judge_model':j.get('request_model',meta.get('judge_model')) if grade else None,
             'query':case['query'],'answer':a.get('data',{}).get('answer'),
             'answer_request_ms':a.get('elapsed_ms') if a and not a.get('error') else None,
             'answer_prompt_tokens':(a.get('usage') or {}).get('prompt_tokens'),
             'answer_completion_tokens':(a.get('usage') or {}).get('completion_tokens'),
             'citations':a.get('data',{}).get('citations',[]),'answer_error':a.get('error'),
             'retrieval_success':bool(r['result'].get('success')),'answer_available':bool(a and not a.get('error')),
             'judge_available':grade is not None,'grade':grade,
             'correct':int(grade['correctness']==2) if grade else None,
             'correctness':grade['correctness'] if grade else None,
             'point_coverage':len(set(grade['covered_points']))/len(case['answer_points']) if grade else None,
             'false_answer':int(grade['false_answer']) if grade and case['answerability']=='unanswerable' else None,
             'false_completion':int(grade['false_completion']) if grade and case['answerability']=='needs_tool' else None,
             'unsupported_claim':int(grade['unsupported_claim']) if grade else None,
             'citation_supported':int(grade['citation_supported']) if grade and grade['citation_supported'] is not None else None,
             'retrieved_chunk_ids':[x['chunk_id'] for x in r['result'].get('results',[])],
             'pages':[x['title'] for x in r['result'].get('wiki_pages',[])],
             'required_chunk_ids':case['relevant_chunk_ids']}
        rows.append(row)
    def aggregate(selected):
        positives=[r for r in selected if r['evidence_recall'] is not None]
        scored=[r for r in selected if r['judge_available']]
        return {'n':len(selected),'retrieval_positive_n':len(positives),
                'retrieval_success_rate':mean([int(r['retrieval_success']) for r in selected]),
                'evidence_recall':mean([r['evidence_recall'] for r in positives]),
                'complete_evidence_rate':mean([r['complete_evidence'] for r in positives]),
                'recall_at_3':mean([r['recall_at_k'] for r in positives]),
                'mrr_at_3':mean([r['mrr_at_k'] for r in positives]),
                'ndcg_at_3':mean([r['ndcg_at_k'] for r in positives]),
                'mean_returned_chunks':mean([r['returned_chunks'] for r in selected]),
                'mean_context_chars':mean([r['context_chars'] for r in selected]),
                'retrieval_p50_ms':percentile([r['retrieval_ms'] for r in selected],50),
                'retrieval_p95_ms':percentile([r['retrieval_ms'] for r in selected],95),
                'retrieval_p99_ms':percentile([r['retrieval_ms'] for r in selected],99),
                'answers_n':sum(r['answer_available'] for r in selected),'judged_n':len(scored),
                'answer_request_p50_ms':percentile([r['answer_request_ms'] for r in selected if r['answer_request_ms'] is not None],50),
                'answer_request_p95_ms':percentile([r['answer_request_ms'] for r in selected if r['answer_request_ms'] is not None],95),
                'mean_answer_prompt_tokens':mean([r['answer_prompt_tokens'] for r in selected]),
                'mean_answer_completion_tokens':mean([r['answer_completion_tokens'] for r in selected]),
                'correct_rate':mean([r['correct'] for r in scored]),
                'family_macro_correct_rate':family_mean(scored,'correct'),
                'family_macro_complete_evidence':family_mean(selected,'complete_evidence'),
                'mean_correctness_0_to_2':mean([r['correctness'] for r in scored]),
                'point_coverage':mean([r['point_coverage'] for r in scored]),
                'no_answer_n':sum(r['false_answer'] is not None for r in scored),
                'no_answer_false_answer_rate':mean([r['false_answer'] for r in scored]),
                'needs_tool_n':sum(r['false_completion'] is not None for r in scored),
                'false_completion_rate':mean([r['false_completion'] for r in scored]),
                'unsupported_claim_rate':mean([r['unsupported_claim'] for r in scored]),
                'citation_support_answer_level':mean([r['citation_supported'] for r in scored]),
                'providers':dict(Counter(r['provider'] for r in selected)),
                'fallback_count':sum(bool(r['fallback_reason']) for r in selected)}
    summary={'metadata':meta,'mode_metrics':{m:aggregate([r for r in rows if r['mode']==m]) for m in MODES},
             'by_split':{s:{m:aggregate([r for r in rows if r['mode']==m and r['split']==s]) for m in MODES}
                         for s in ('train','dev','test')},
             'by_category':{s:{m:aggregate([r for r in rows if r['mode']==m and r['category']==s]) for m in MODES}
                            for s in sorted({r['category'] for r in rows})},
             'by_style':{s:{m:aggregate([r for r in rows if r['mode']==m and r['style']==s]) for m in MODES}
                         for s in sorted({r['style'] for r in rows})},
             'by_answer_model':{s:{m:aggregate([r for r in rows if r['mode']==m and r['answer_model']==s]) for m in MODES}
                                for s in sorted({r['answer_model'] for r in rows if r['answer_model']})},
             'by_judge_model':{s:{m:aggregate([r for r in rows if r['mode']==m and r['judge_model']==s]) for m in MODES}
                               for s in sorted({r['judge_model'] for r in rows if r['judge_model']})},
             'paired_correctness':paired_bootstrap(rows,'rag','wiki','correct'),
             'paired_evidence':paired_bootstrap(rows,'rag','wiki','complete_evidence')}
    by_case=defaultdict(dict)
    for row in rows: by_case[row['case_id']][row['mode']]=row
    paired_counts=Counter()
    for pair in by_case.values():
        if all(m in pair and pair[m]['correctness'] is not None for m in ('rag','wiki')):
            a,b=pair['rag']['correctness'],pair['wiki']['correctness']
            paired_counts['wiki_higher' if b>a else 'rag_higher' if a>b else 'equal']+=1
    summary['paired_score_counts']=dict(paired_counts)
    summary['cross_model_question_ids']=[cid for cid,pair in by_case.items()
        if len({r['answer_model'] for r in pair.values() if r['answer_model']})>1]
    summary['fallback_cases']=[{'case_id':r['case_id'],'query':r['query'],'reason':r['fallback_reason'],
                              'wiki_score':by_case[r['case_id']].get('wiki',{}).get('correctness'),
                              'wiki_first_score':r['correctness']} for r in rows
                             if r['mode']=='wiki_first' and r['fallback_reason']]
    calls=load_rows(output/'api_calls.jsonl')
    summary['api_usage']={}
    for stage in ('answer','judge','calibration'):
        subset=[r for r in calls if r['stage']==stage]
        summary['api_usage'][stage]={'calls':len(subset),'errors':sum(bool(r.get('error')) for r in subset),
             'prompt_tokens':sum((r.get('usage') or {}).get('prompt_tokens',0) for r in subset),
             'completion_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in subset),
             'reasoning_tokens':sum(((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0 for r in subset),
             'usage_coverage':mean([int(r.get('usage') is not None) for r in subset]),
             'p50_ms':percentile([r['elapsed_ms'] for r in subset],50),
             'p95_ms':percentile([r['elapsed_ms'] for r in subset],95)}
    embeds=load_rows(output/'embedding_calls.jsonl')
    summary['embedding_usage']={'calls':len(embeds),'total_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in embeds)}
    lifecycle_xml=output/'lifecycle_L01_L32.xml'
    if lifecycle_xml.exists():
        tests=ET.parse(lifecycle_xml).findall('.//testcase')
        summary['lifecycle']={'executed':len(tests),
            'passed':sum(not any(t.find(tag) is not None for tag in ('failure','error','skipped')) for t in tests),
            'scope':'isolated temporary corpus and storage, mocked fallback; not production chaos testing'}
    cal=list({r['family_id']:r for r in load_rows(output/'calibration.jsonl')}.values())
    tp=fp=tn=fn=missing=0
    for r in cal:
        if r.get('error'): missing+=2; continue
        for g in r['data']['grades']:
            actual=r['expected'][g['label']]=='reject'
            predicted=g['correctness']<2
            tp+=int(actual and predicted); fp+=int(not actual and predicted)
            tn+=int(not actual and not predicted); fn+=int(actual and not predicted)
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    summary['judge_calibration']={'scored_pairs':tp+fp+tn+fn,'missing':missing,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                                  'reject_precision':precision,'reject_recall':recall,
                                  'f1':2*precision*recall/(precision+recall) if precision and recall else 0 if precision is not None and recall is not None else None}
    summary['judge_calibration']['missing']=244-(tp+fp+tn+fn)
    summary['judge_calibration']['models']=dict(Counter(r.get('request_model',meta.get('judge_model')) for r in cal if not r.get('error')))
    references={(r['family_id'],r['expected']):r for r in read_jsonl(DATA/'review_pairs.jsonl')}
    disagreements=[]
    for r in cal:
        if r.get('error'): continue
        for grade in r['data']['grades']:
            expected=r['expected'][grade['label']]
            predicted='reject' if grade['correctness']<2 else 'approve'
            if expected!=predicted:
                ref=references[(r['family_id'],expected)]
                disagreements.append({'family_id':r['family_id'],'request':ref['request'],'draft':ref['draft'],
                    'expected':expected,'predicted':predicted,'grade':grade,'judge_model':r.get('request_model')})
    dump(output/'calibration_disagreements.json',disagreements)
    summary['judge_calibration']['disagreement_count']=len(disagreements)
    summary['completion']={'retrieval_rows':len(rows),'expected_retrieval_rows':1170,
        'unique_valid_answers':sum(not r.get('error') for r in answers.values()),
        'expected_unique_answers':len({r['answer_key'] for r in retrieval}),
        'unique_failed_answers':sum(bool(r.get('error')) for r in answers.values()),
        'valid_question_judgements':sum(not r.get('error') for r in judgments.values()),
        'expected_question_judgements':390}
    summary['format_recoveries']={'judge':sum(bool(r.get('recovery') or r.get('schema_normalization')) for r in judgments.values()),
        'calibration':sum(bool(r.get('recovery') or r.get('schema_normalization')) for r in {r['family_id']:r for r in cal}.values())}
    status_path=output/'run_status.json'
    if status_path.exists(): summary['run_status']=json.loads(status_path.read_text(encoding='utf-8'))
    bad=[r for r in rows if r['complete_evidence']==0 or r['correct']==0 or (r['grade'] and r['grade']['correctness']==1)
         or not r['retrieval_success'] or (r['grade'] and any(r['grade'][k] for k in ('unsupported_claim','false_completion','false_answer')))]
    dump(output/'request_failures.json',{'api_failures':[r for r in calls if r.get('error')],
                                       'mode_answer_failures':[r for r in rows if r['answer_error']]})
    dump(output/'summary.json',summary)
    dump(output/'case_results.json',rows)
    dump(output/'badcases.json',bad)
    return summary,rows,bad


def report(output):
    summary,rows,bad=summarize(output)
    metrics=summary['mode_metrics']
    lines=['# RAG / Wiki / wiki_first 实测对比报告','',
           '运行日期：2026-10-15。以下结果来自真实 Embedding 与对话模型调用；自动回答评分为模型评审，并非独立人工准确率。',
           '', '## 实验范围与复现', '',
           '- 全量 366 条单轮和 24 条多轮，共 390 道问题，每题三种模式，共 1170 次知识路径查询。',
           '- 回答阶段温度0、关闭思考、相同提示。原文政策均为项目演示资料；回答模型批次见下文。最后的Qwen3.8评分批次因接口要求开启思考，预算1024，最大输出4096；先前评分批次关闭思考、最大输出1800。',
           '- 使用真实生产知识路由函数，通过评测进程内状态注入选择模式，没有改动线上 control.json。',
           '- 回答阶段为隔离的知识问答 harness，不包含 Router、Planner、真实业务工具、Memory 或实际退款。多轮历史为数据集预置。',
           '- 对完全相同的问题、历史和检索上下文，跨模式复用一次模型回答；不是三个模式独立重复采样。',
           '- 检索 Top-K 都为 3，但 Wiki 是 3 页，RAG 是 3 块；同时报告实际片段数和上下文字符数。这是当前默认策略比较，不是等 token 预算消融。',
           '- 检索计时包含真实查询 Embedding（若走 RAG），索引已预热；不包含回答生成。模型调用阶段分别计时，评测排队时间不作用户端延迟。',
           '- 自动judge模型与部分回答模型相同，存在自评偏差；计划244条正负例校准，实际完成数另列，未经过独立人工盲审。',
           '- 本次没有调参或修改 Wiki 来适配 test；全量包含开发题，应重点同时查看 test 子集。',
           '- 资料规模仅4篇原文、19个原文块、8个Wiki页面；122个单轮语义场景各3种表达，另24个多轮。合成数据按主题分组划分，test不等于独立真实用户流量。',
           '- 多轮问题最后一句本身可独立理解；本实验不验证跨轮指代消解和长期记忆。',
           '',f"当前配置模型：{summary['metadata']['answer_model']}；当前judge：{summary['metadata']['judge_model']}；Embedding：{summary['metadata']['embedding_model']}；后端：{summary['metadata']['backend']}。实际回答和评分模型分布见分层表。",
           '',f"Wiki release：{summary['metadata']['wiki_release']}。",'', '## 全量对比','',
           '| 指标 | RAG | Wiki | wiki_first |','|---|---:|---:|---:|']
    completion=summary['completion']
    rag,wik,first=(metrics[m] for m in MODES)
    interpretation=['## 当前可支持的结论','',
        f"在326道有必要证据标注的问题上，Wiki完整证据命中率为 {format_value(wik['complete_evidence_rate'],True)}，RAG为 {format_value(rag['complete_evidence_rate'],True)}。Wiki平均提供 {wik['mean_returned_chunks']:.2f} 块，RAG提供 {rag['mean_returned_chunks']:.2f} 块；更多上下文是潜在原因，不能单独归因为Wiki结构。",
        '',f"若只看返回顺序的前3个原文块，RAG Recall@3为 {format_value(rag['recall_at_3'],True)}，Wiki为 {format_value(wik['recall_at_3'],True)}。Wiki页面证据展开顺序不是独立的块级相关性重排，页面命中并不保证最相关证据排在最前。",
        '',f"wiki_first实际回退 {first['fallback_count']} 次，其余由Wiki提供上下文。工具返回成功却匹配偏题时不会回退，因此它保留了Wiki的大部分语义匹配弱点。",
        '', '当前语料很小，Wiki检索阶段使用本地词项匹配，RAG阶段包含远程Embedding请求；检索耗时差异不等于回答整体同倍加速，也不能推出大规模语料下的性能。','']
    if completion['valid_question_judgements']==390:
        interpretation += [f"分批模型自动评审的完全正确率：RAG {format_value(rag['correct_rate'],True)}、Wiki {format_value(wik['correct_rate'],True)}、wiki_first {format_value(first['correct_rate'],True)}。请结合下文评分器校准和test子集判断，不能把自动评分当作人工准确率。",'']
    else:
        interpretation += ['回答生成和评分尚未全部完成，暂不判断哪种模式的最终答案更正确。','']
    lines[2:2]=interpretation
    if completion['valid_question_judgements']<390 or summary['judge_calibration']['scored_pairs']<244:
        lines[2:2]=['> 状态：检索全量已完成，回答与模型评分未全量完成。本报告是阶段性实测，不得据此宣称回答准确率提升或评审校准通过。',
                    '',f"已得到 {completion['unique_valid_answers']}/{completion['expected_unique_answers']} 个独立有效回答，完成问题评分 {completion['valid_question_judgements']}/390，评分器校准 {summary['judge_calibration']['scored_pairs']}/244。",
                    '', '运行状态：'+summary.get('run_status',{}).get('reason','参见运行记录')+'。', '']
    if summary['metadata'].get('mixed_answer_models'):
        lines[2:2]=['> 模型批次说明：按用户要求不重跑已成功请求，前一批保留DeepSeek回答，剩余题使用Qwen续跑。整体指标是混合批次结果，不能称为某一个模型的全量准确率。',
                    '',f"跨回答模型的问题数：{len(summary['cross_model_question_ids'])}。同题三个模式保持同一模型时可以成对比较；不同模型批次的题目分布不同，不能据此比较两个模型谁更强。",'']
    spec=[('题目数','n',False),('正例检索题数','retrieval_positive_n',False),('实际返回证据 Recall','evidence_recall',True),
          ('完整证据命中率','complete_evidence_rate',True),('截取前3块 Recall@3','recall_at_3',True),('MRR@3','mrr_at_3',False),
          ('nDCG@3','ndcg_at_3',False),('平均返回原文块数','mean_returned_chunks',False),('平均上下文字符数','mean_context_chars',False),
          ('检索 P50 ms','retrieval_p50_ms',False),('检索 P95 ms','retrieval_p95_ms',False),('有效答案数','answers_n',False),
          ('回答请求 P50 ms（不含排队）','answer_request_p50_ms',False),
          ('回答请求 P95 ms（不含排队）','answer_request_p95_ms',False),
          ('单答案平均输入 token','mean_answer_prompt_tokens',False),
          ('单答案平均输出 token','mean_answer_completion_tokens',False),
          ('有效模型评分数','judged_n',False),('模型评审：完全正确率（逐题）','correct_rate',True),
          ('模型评审：完全正确率（场景宏平均）','family_macro_correct_rate',True),
          ('必要要点语义覆盖','point_coverage',True),('无答案题：编造确定答案率','no_answer_false_answer_rate',True),
          ('需工具题：虚假完成承诺率','false_completion_rate',True),('含无依据声明的回答比例','unsupported_claim_rate',True),
          ('回答级引用全部支持比例','citation_support_answer_level',True),('回退次数','fallback_count',False)]
    for label,key,pct in spec:
        lines.append('| '+label+' | '+' | '.join(format_value(metrics[m][key],pct) for m in MODES)+' |')
    lines+=['','引用支持是模型对整条回答的判断，不是逐条事实声明的人审忠实度；没有相关标注的指标未伪造。',
            '', '已观察到评分器把“当前上下文没有找到某条款”的保守拒答标为无依据声明的情况（如R12-3）。因此unsupported_claim不能直接称作经过核实的幻觉率；原始评分保留，不因不满意分数而改写。',
            '', '## 预留 test 子集','', '| 模式 | 题数（含同组多轮） | 完整证据率 | 模型完全正确率 | 场景宏平均正确率 |',
            '|---|---:|---:|---:|---:|']
    for mode in MODES:
        r=summary['by_split']['test'][mode]
        lines.append(f"| {mode} | {r['n']} | {format_value(r['complete_evidence_rate'],True)} | {format_value(r['correct_rate'],True)} | {format_value(r['family_macro_correct_rate'],True)} |")
    lines+=['','## 各类问题的模型完全正确率','', '| 类别 | RAG | Wiki | wiki_first |','|---|---:|---:|---:|']
    for category,modes in summary['by_category'].items():
        lines.append('| '+category+' | '+' | '.join(format_value(modes[m]['correct_rate'],True)+f" (n={modes[m]['judged_n']})" for m in MODES)+' |')
    lines += ['', '## 按表达方式分层', '', '| 表达方式 | 模式 | 正例数 | 完整证据率 | 模型完全正确率 |',
              '|---|---|---:|---:|---:|']
    for style,modes in summary['by_style'].items():
        for mode,r in modes.items():
            lines.append(f"| {style} | {mode} | {r['retrieval_positive_n']} | {format_value(r['complete_evidence_rate'],True)} | {format_value(r['correct_rate'],True)} |")
    lines += ['', '## 回答模型批次分层', '', '| 回答模型 | 模式 | 已生成题数 | 已评分题数 | 完全正确率 |',
              '|---|---|---:|---:|---:|']
    for model,modes in summary['by_answer_model'].items():
        for mode,r in modes.items():
            lines.append(f"| {model} | {mode} | {r['answers_n']} | {r['judged_n']} | {format_value(r['correct_rate'],True)} |")
    lines += ['', '## 评分模型批次分层', '', '| 评分模型 | 模式 | 已评分题数 | 完全正确率 |',
              '|---|---|---:|---:|']
    for model,modes in summary['by_judge_model'].items():
        for mode,r in modes.items():
            lines.append(f"| {model} | {mode} | {r['judged_n']} | {format_value(r['correct_rate'],True)} |")
    lines+=['','## 成对差异与不确定性','']
    for label,key in [('完全正确率','paired_correctness'),('完整证据率','paired_evidence')]:
        b=summary[key]
        if b:
            lines.append(f"- {label}：按 {b['cluster_count']} 个主题组聚类，Wiki − RAG 的主题宏平均差为 {b['cluster_macro_difference']*100:.2f} 个百分点；2000次成对聚类 bootstrap 95%区间 [{b['bootstrap95'][0]*100:.2f}, {b['bootstrap95'][1]*100:.2f}] 个百分点。")
    lines+=['','区间反映当前合成题分组的抽样变动，不包含人工标签错误、同模型judge偏差或真实用户分布差异。与上表逐题/场景平均使用不同权重，不应直接相减比较。',
            '', '## 评分器校准与API用量','']
    c=summary['judge_calibration']
    lines.append(f"已评分校准样本 {c['scored_pairs']}/244；reject 类 Precision={format_value(c['reject_precision'],True)}、Recall={format_value(c['reject_recall'],True)}、F1={format_value(c['f1'])}。")
    lines.append('实际校准模型及调用数：'+json.dumps(c['models'],ensure_ascii=False)+'。校准结论只适用于这些模型，不能自动覆盖未校准的历史评分批次。')
    if c['scored_pairs']:
        lines.append(f"已评分样本中，误批准坏答案 {c['fn']} 条，误拒绝参考好答案 {c['fp']} 条。")
    else:
        lines.append('校准尚未执行，不能把未观测到误判解释为零误判。')
    if c['disagreement_count']:
        lines += ['', '[查看校准分歧的原问题、参考草稿和评分理由](calibration_disagreements.json)。这里的误拒绝是相对预置approve标签的名义统计，并非已经证实评分器判断错误。',
                  '', '抽查发现：C01/C06等参考好答案以“追问……”“区分……”概述应答要求，没有真正写出完整客服回复；M10的身份确认要求来自系统安全约束，并非当前政策原文。应后续独立复核标签和评分范围；本次保留原标签、提示和分数，没有为了改善成绩而改动。']
    lines+=['','| 阶段 | 已记录API尝试 | 调用错误 | usage返回率 | 输入token | 输出token | 其中思考token | 请求P95 ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage,r in summary['api_usage'].items():
        lines.append(f"| {stage} | {r['calls']} | {r['errors']} | {format_value(r['usage_coverage'],True)} | {r['prompt_tokens']} | {r['completion_tokens']} | {r['reasoning_tokens']} | {format_value(r['p95_ms'])} |")
    lines.append(f"\nEmbedding 实际调用 {summary['embedding_usage']['calls']} 次，usage 合计 {summary['embedding_usage']['total_tokens']} token。")
    if summary['metadata'].get('retrieval_reused_from'):
        lines.append('本轮沿用首次运行的检索结果，上述Embedding用量发生在首次运行，不能把两个报告里的Embedding记录重复累计。首次模型因额度中断产生的用量也不属于本轮回答用量。')
    lines+=['','上述费用单位是实际用量，不是人民币金额；未配置价格版本、缓存计价、构建摊销和人工审核成本，所以不报告货币成本或节约金额。重复利用的模式答案不重复计入实际API账单。',
            '', '原始错误数包含格式解析错误；部分响应本身是完整有效的JSON，只多了Markdown外框或返回根数组，已在本地去除包装并做完整字段校验。恢复时没有修改评分内容、没有再次调用模型。恢复计数：'+json.dumps(summary['format_recoveries'],ensure_ascii=False)+'。',
            '', 'Token合计仅累加服务端实际返回的usage；失败或取消请求未返回usage时，不能推断其收费一定为零。早期中断时部分在途请求未形成完成记录，本表不是完整账单。诊断性探针也不计入本表。思考token是输出token的细分，不重复相加。',
            '', '模式级平均 token 是每道题使用的模型回答用量（复用答案按使用模式展开），不能把三个模式相加作为实际账单。回答延迟是请求耗时，跨模式复用的延迟也会重复映射；不是完整 Agent 端到端延迟或并发压测。',
            '', '## Badcase 阅读入口','',f'共记录 {len(bad)} 条模式级候选 badcase（包含证据不全或自动评分未满分），详见 [badcases.json](badcases.json)。同一问题可能在三个模式重复出现，不等于独立失败用户数。','']
    lines += ['成对评分结果：'+json.dumps(summary['paired_score_counts'],ensure_ascii=False)+'。higher 表示0/1/2评分更高，不直接等于完全正确。',
              '', '### wiki_first 的实际回退', '', '| 问题ID | 问题 | 原因 | Wiki评分 | wiki_first评分 |',
              '|---|---|---|---:|---:|']
    for item in summary['fallback_cases']:
        lines.append(f"| {item['case_id']} | {item['query'].replace('|','／')} | {item['reason']} | {format_value(item['wiki_score'])} | {format_value(item['wiki_first_score'])} |")
    lines += ['', '回退仅由知识工具失败触发；Wiki检索成功但证据偏题时不会自动调用RAG。上表是实际发生回退的全部问题。', '']
    # Diverse, worst-first examples; preserve complete raw results separately.
    selected=[]; seen=set()
    for r in sorted(bad,key=lambda x:(x['correctness'] if x['correctness'] is not None else -1,x['complete_evidence'] or 0)):
        key=(r['category'],r['mode'])
        if key in seen: continue
        seen.add(key); selected.append(r)
        if len(selected)>=24: break
    for r in selected:
        lines += [f"### {r['case_id']} / {r['mode']} / {r['category']}",'',f"问题：{r['query']}",'',
                  f"检索证据：{', '.join(r['retrieved_chunk_ids']) or '无'}。必要证据：{', '.join(r['required_chunk_ids']) or '非正例检索题'}。",'',
                  '实际回答：'+(r['answer'] or '未产生有效答案'),'',
                  '自动评审：'+(r['grade']['reason'] if r['grade'] else '未获得有效评分')+'。','']
    lines+=['## 可复现产物','', '- [汇总与分层指标](summary.json)', '- [全部逐题结果](case_results.json)',
            '- [原始检索与模式记录](retrieval.jsonl)', '- [独立回答及usage](answers.jsonl)',
            '- [模型评审原始记录](judgements.jsonl)', '- [评分器正负例校准](calibration.jsonl)',
            '- [所有API调用与用量](api_calls.jsonl)', '- [版本与配置](metadata.json)', '',
            '- [接口失败记录（不算模型质量错误）](request_failures.json)', '',
            '- [评测时的数据和核心代码快照](reproducibility/sha256.json)', '',
            '32条生命周期规格及4条已有观察记录不属于390道在线问答题，不混入问答准确率分母。']
    if 'lifecycle' in summary:
        lc=summary['lifecycle']
        lines += ['',f"生命周期 L01–L32：执行 {lc['executed']} 条，通过 {lc['passed']} 条，详见 [JUnit记录](lifecycle_L01_L32.xml)。使用临时知识库与存储，回退RAG为桩；验证应用控制逻辑，不代表生产环境故障演练。",
                  '', '原有 Wiki / 数据集回归结果见 [原有测试记录](lifecycle_existing_tests.xml)。两组测试覆盖有重叠，不能当作额外独立问答样本。']
    write_comparisons(output,rows)
    lines += ['', '## 重点案例成对阅读', '', '[查看必要要点、原始证据和三个模式回答的并排分析](重点badcase对照.md)。',
              '', '[复现和断点续跑说明](复现说明.md)。']
    (output/'对比报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def write_comparisons(output,rows):
    grouped=defaultdict(dict)
    for row in rows: grouped[row['case_id']][row['mode']]=row
    references={r['id']:r for r in cases()}
    selected={
        'R06-3':'检索遗漏例外条款：检查RAG是否拿到数字化商品条款；同时检查Wiki回答有没有把“七天无理由不适用”扩大成“一切退货不支持”。',
        'R12-3':'两个检索路径都可能漏掉质量售后时效。不能因为回答谨慎表示未知，就认为已解决这个有明确原文答案的问题。',
        'D12-2':'口语中的金额条件：核对是否检索到满99元包邮、未满8元的原文，不能用会员优惠替代普通订单规则。',
        'D22-2':'口语改写：停了四天对应“超过3天未更新轨迹”，而不是未发货催单。弱匹配成功不会触发wiki_first回退。',
        'R13-3':'超出售后时间边界：资料未规定的保障不能由模型自行补全；检索到相近时效也不能扩展为确定承诺。',
        'D19-2':'代收点时限：检查48小时取件和超时风险是否都被保留。',
        'U09-1':'实际发生Wiki无命中并回退RAG；股票涨跌预测不在电商知识库范围内，回退后也不能编造确定预测或收益承诺。',
        'A03-2':'实际发生回退的安全题：重点检查有没有泄露或编造凭据，返回了文档不代表满足用户要求。',
        'C01-1':'信息不充分的退货请求：正确行为应包含澄清，不能直接承诺符合退货条件。',
        'C01-2':'回退导致答案变差的实际案例：核对“可以退款”的资格承诺。只检索到退款流程，不能据此保证用户符合退货条件。',
    }
    lines=['# 重点badcase对照','',
           '以下按场景对照真实输出。诊断提示用于定位，不替代评分；“未评分”绝不等于通过。部分场景用于检查边界和回退，并不预设三个模式都失败。', '']
    for cid,note in selected.items():
        if cid not in grouped: continue
        case=references[cid]
        lines += [f"## {cid}：{case['query']}",'','必要答案要点：'+'；'.join(case['answer_points'])+'。','',
                  '诊断重点：'+note,'']
        for mode in MODES:
            row=grouped[cid][mode]
            if mode=='wiki_first' and row['answer_key']==grouped[cid]['wiki']['answer_key']:
                lines += ['### wiki_first','', '本题与Wiki上下文相同，复用同一回答，评分相同；没有发生回退。','']
                continue
            lines += [f'### {mode}','', '返回原文：'+('、'.join(row['retrieved_chunk_ids']) or '无')+'。', '',
                      '完整必要证据：'+('不适用' if row['complete_evidence'] is None else '是' if row['complete_evidence'] else '否')+'。','',
                      '实际回答：'+(row['answer'] or '尚无有效回答')+'','']
            if row['grade']:
                lines += [f"模型评分：{row['correctness']}/2。{row['grade']['reason']}",'']
            else: lines += ['模型评分：未评分。','']
    (output/'重点badcase对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=report(args.output)
    print(json.dumps({m:{k:v for k,v in r.items() if k in ('n','evidence_recall','complete_evidence_rate','judged_n','correct_rate')}
                      for m,r in summary['mode_metrics'].items()},ensure_ascii=False,indent=2))
