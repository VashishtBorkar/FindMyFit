"""Select high-value experiment finalists on validation, then score test once."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from findmyfit.models.metric import FashionCompatibilityModel
from training.metric_learning.high_value_experiments import load_data, build_model, evaluate_full
from training.metric_learning.ranking_data import RankingDataset
from training.metric_learning.compare_retrieval import evaluate, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args();out=args.runs
    reference=out.parent/'controlled-seed-42'
    torch.set_num_threads(4)
    embeddings,examples,context,reference_report,_=load_data(reference)
    category_ids=reference_report['category_ids']
    selected=json.loads((out/'replication_selection.json').read_text())['selected_on_validation_only']
    names=['baseline_ranking','baseline_contrastive',*selected]
    records={}
    for name in names:
        records[name]={}
        for seed in (42,43,44):
            if name.startswith('baseline'):
                key='shared_projector' if name=='baseline_ranking' else 'shared_contrastive'
                directory=out.parent/f'controlled-seed-{seed}'
                r=json.loads((directory/'metrics.json').read_text())
                assert r['candidates_sha256']==reference_report['candidates_sha256']
                assert r['vector_sha256']==reference_report['vector_sha256']
                r=r['models'][key];checkpoint=directory/f'{key}.pt'
            else:
                directory=out/f'{name}-seed-{seed}';r=json.loads((directory/'metrics.json').read_text());checkpoint=directory/'model.pt'
            records[name][seed]={'record':r,'checkpoint':str(checkpoint)}
    def validation(r):
        return r['history'][r['best_epoch']-1]['validation']['mrr']
    selection={name:statistics.mean(validation(r['record']) for r in seeds.values()) for name,seeds in records.items()}
    preferred=max(selection,key=selection.get)
    # No new test predictions have been computed before this selection is persisted.
    write_json(out/'final_selection.json',{'preferred_by_mean_validation_mrr':preferred,'mean_validation_mrr':selection,'seeds':[42,43,44],'test_used_for_selection':False})
    print('SELECTION',preferred,selection,flush=True)
    summary={};all_scores={}
    for name,seeds in records.items():
        scores={}
        for seed,item in seeds.items():
            r=item['record'];cp=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
            if name.startswith('baseline'):
                model=FashionCompatibilityModel(**cp['model_kwargs'])
            else:
                model,_=build_model(cp['config'],category_ids)
            model.load_state_dict(cp['model_state_dict']);model.eval()
            loader=DataLoader(RankingDataset(examples['validation'],embeddings,category_ids),batch_size=64)
            replay=evaluate(model,loader,torch.device('cpu'))
            assert abs(replay['mrr']-validation(r))<1e-7
            full_validation=r.get('full_validation') or evaluate_full(model,embeddings,context['validation'],examples['validation'],category_ids)
            sampled=evaluate(model,DataLoader(RankingDataset(examples['test'],embeddings,category_ids),batch_size=64),torch.device('cpu'))
            if name.startswith('baseline'):
                assert abs(sampled['mrr']-r['test']['mrr'])<1e-7
            full=evaluate_full(model,embeddings,context['test'],examples['test'],category_ids)
            scores[seed]={'sampled_test':sampled,'full_test':full,'sampled_validation':replay,'full_validation':full_validation,'best_epoch':r['best_epoch'],'epochs_run':len(r['history']),'training_seconds':sum(h['seconds'] for h in r['history']),'parameters':r['parameters']}
            print('SCORED',name,seed,'R5',sampled['recall_at_5'],'fullMRR',full['mrr'],flush=True)
        all_scores[name]=scores
        def aggregate(path):
            vals=[]
            for r in scores.values():
                for k in path:r=r[k]
                vals.append(r)
            return {'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals)}
        summary[name]={'parameters':next(iter(scores.values()))['parameters'],'mean_validation_mrr':selection[name],'sampled_test':{k:aggregate(['sampled_test',k]) for k in ('recall_at_5','recall_at_10','mrr')},'full_test':{k:aggregate(['full_test',k]) for k in ('hit_at_5','recall_at_5','mrr','ndcg_at_5')},'full_validation_mrr':aggregate(['full_validation','mrr']),'mean_training_seconds':statistics.mean(r['training_seconds'] for r in scores.values()),'by_category':{cat:{'sampled_recall_at_5':aggregate(['sampled_test','by_category_id',str(category_ids[cat]),'recall_at_5']),'full_mrr':aggregate(['full_test','by_category',cat,'mrr'])} for cat in ('pants','shoes')}}
        write_json(out/'comparison.json',{'preferred_by_mean_validation_mrr':preferred,'summary':summary,'runs':all_scores})
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['configuration','seed','parameters','best_epoch','epochs_run','training_seconds','sampled_recall5','sampled_mrr','full_hit5','full_recall5','full_mrr'])
        for name,seeds in all_scores.items():
            for seed,r in seeds.items():w.writerow([name,seed,r['parameters'],r['best_epoch'],r['epochs_run'],r['training_seconds'],r['sampled_test']['recall_at_5'],r['sampled_test']['mrr'],r['full_test']['hit_at_5'],r['full_test']['recall_at_5'],r['full_test']['mrr']])
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
