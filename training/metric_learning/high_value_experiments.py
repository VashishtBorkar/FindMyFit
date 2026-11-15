"""Controlled resampling, tuning, and category-head experiments over frozen features."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.storage.local_images import LocalImageStore
from findmyfit.models.metric import FashionCompatibilityModel
from findmyfit.models.category_heads import CategoryHeadCompatibilityModel
from training.metric_learning.data import load_embeddings
from training.metric_learning.ranking_data import RankingDataset, ResampledRankingDataset, make_ranking_splits
from training.metric_learning.compare_retrieval import train, evaluate, write_json, check_training_batch
from training.metric_learning.loss import CandidateRankingLoss, CandidateContrastiveLoss

CONFIGS = {
    'ranking_resample': dict(objective='ranking', resample=True),
    'contrastive_resample': dict(objective='contrastive', resample=True),
    'ranking_lr0003': dict(objective='ranking', resample=True, learning_rate=0.0003),
    'ranking_tau005': dict(objective='ranking', resample=True, temperature=0.05),
    'ranking_tau02': dict(objective='ranking', resample=True, temperature=0.2),
    'contrastive_lr0003': dict(objective='contrastive', resample=True, learning_rate=0.0003),
    'contrastive_margin05': dict(objective='contrastive', resample=True, margin=0.5),
    'contrastive_margin15': dict(objective='contrastive', resample=True, margin=1.5),
    'heads_fixed': dict(objective='ranking', resample=False, category_heads=True),
    'heads_resample': dict(objective='ranking', resample=True, category_heads=True),
}


def build_model(config, category_ids):
    if config.get('category_heads'):
        kwargs = dict(embedding_dim=512, category_ids=[category_ids[c] for c in ('pants', 'shoes')])
        return CategoryHeadCompatibilityModel(**kwargs), kwargs
    return FashionCompatibilityModel(512), dict(embedding_dim=512)


@torch.no_grad()
def evaluate_full(model, embeddings, context, examples, category_ids):
    """Rank all canonical split items, crediting every known partner per anchor/category."""
    model.eval()
    groups = sorted({(row[0], row[1]) for row in examples})
    per_category = {}
    for category in sorted({c for _, c in groups}):
        pool = context['pools'][category]
        index = {item: i for i, item in enumerate(pool)}
        anchors = [a for a, c in groups if c == category]
        cid = category_ids[category]
        def encode(ids):
            chunks=[]
            for start in range(0, len(ids), 256):
                x=torch.stack([torch.from_numpy(embeddings[i]['embedding']) for i in ids[start:start+256]])
                if isinstance(model, CategoryHeadCompatibilityModel):
                    x=model.encode(x,torch.full((len(x),),cid,dtype=torch.long))
                else:
                    x=model(x)
                chunks.append(x)
            return torch.cat(chunks)
        items=encode(pool); queries=encode(anchors)
        rows=[]
        for start in range(0,len(anchors),64):
            scores=queries[start:start+64] @ items.T
            order=scores.argsort(dim=1,descending=True,stable=True)
            for anchor,ordered in zip(anchors[start:start+64],order):
                relevant={index[i] for i in context['partners'][anchor] if i in index}
                ranks=[j+1 for j,i in enumerate(ordered.tolist()) if i in relevant]
                assert ranks
                hits=sum(r<=5 for r in ranks)
                dcg=sum(1/torch.log2(torch.tensor(float(r+1))).item() for r in ranks if r<=5)
                ideal=sum(1/torch.log2(torch.tensor(float(r+1))).item() for r in range(1,min(5,len(ranks))+1))
                rows.append(dict(mrr=1/min(ranks),hit_at_5=float(min(ranks)<=5),recall_at_5=hits/len(ranks),ndcg_at_5=dcg/ideal))
        per_category[category]={'queries':len(rows),'catalog_items':len(pool),**{k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}}
    total=sum(r['queries'] for r in per_category.values())
    return {'queries':total,**{k:sum(r[k]*r['queries'] for r in per_category.values())/total for k in ('mrr','hit_at_5','recall_at_5','ndcg_at_5')},'by_category':per_category}


def load_data(reference):
    settings=Settings.from_env()
    repo=SqliteEmbeddingRepository(create_session_factory(settings.database_url),LocalImageStore(settings.images_dir))
    embeddings,_=load_embeddings(repo,model_name=settings.clip_catalog_model_name,model_version=settings.clip_model_version)
    digest=hashlib.sha256()
    for key,record in sorted(embeddings.items()):
        digest.update(key.encode()+b'\0');digest.update(record['embedding'].tobytes())
    report=json.loads((reference/'metrics.json').read_text())
    assert digest.hexdigest()==report['vector_sha256']
    candidates_path=reference/'candidates.json'
    assert hashlib.sha256(candidates_path.read_bytes()).hexdigest()==report['candidates_sha256']
    examples=json.loads(candidates_path.read_text())
    context={}
    generated,audit=make_ranking_splits(settings.project_root/'data',embeddings,seed=42,candidate_count=32,anchor_category='top',target_categories=['pants','shoes'],sampling_context=context)
    # JSON serializes original tuples as lists; compare canonical serialized form.
    assert json.loads(json.dumps(generated))==examples
    return embeddings,examples,context,report,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['screen','replicate'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    root=Settings.from_env().project_root
    reference=root/'data/tracks/fallback/runs/controlled-seed-42'
    embeddings,examples,context,reference_report,audit=load_data(reference)
    if args.stage=='screen':
        args.output.mkdir(exist_ok=False,parents=True)
        source=args.output/'source';source.mkdir()
        for path in [Path(__file__),Path('training/metric_learning/compare_retrieval.py'),Path('training/metric_learning/ranking_data.py'),Path('training/metric_learning/loss.py'),Path('src/findmyfit/models/metric.py'),Path('src/findmyfit/models/category_heads.py')]:
            shutil.copyfile(path,source/path.name)
        write_json(args.output/'protocol.json',dict(configs=CONFIGS,seeds=[42,43,44],screening_seed=42,selection='highest sampled validation MRR within objective/family; replicate default resampling and selected tuning/head configurations',epochs=30,patience=5,batch_size=64,evaluation_reference=str(reference),candidates_sha256=reference_report['candidates_sha256'],vector_sha256=reference_report['vector_sha256'],category_ids=reference_report['category_ids'],audit=audit))
        work=[(name,42) for name in CONFIGS]
    else:
        records=json.loads((args.output/'screen_summary.json').read_text())
        chosen=[]
        for family in [('ranking_resample','ranking_lr0003','ranking_tau005','ranking_tau02'),('contrastive_resample','contrastive_lr0003','contrastive_margin05','contrastive_margin15'),('heads_fixed','heads_resample')]:
            chosen.append(max(family,key=lambda n:records[n]['best_validation_mrr']))
        chosen=list(dict.fromkeys(['ranking_resample','contrastive_resample',*chosen]))
        write_json(args.output/'replication_selection.json',{'selected_on_validation_only':chosen,'screening_results':records})
        work=[(name,seed) for seed in (43,44) for name in chosen]
    for name,seed in work:
        config=dict(learning_rate=0.001,temperature=0.1,margin=1.0,**{})
        config.update(CONFIGS[name])
        directory=args.output/f'{name}-seed-{seed}';directory.mkdir(exist_ok=False)
        category_ids=reference_report['category_ids']
        datasets={split:RankingDataset(rows,embeddings,category_ids) for split,rows in examples.items()}
        if config['resample']:
            datasets['train']=ResampledRankingDataset(examples['train'],embeddings,category_ids,context['train'],seed=9000+seed)
        torch.manual_seed(seed);model,kwargs=build_model(config,category_ids)
        criterion=CandidateContrastiveLoss(config['margin']) if config['objective']=='contrastive' else CandidateRankingLoss(config['temperature'])
        check=check_training_batch(model,datasets['train'],torch.device('cpu'),criterion)
        torch.manual_seed(seed);model,kwargs=build_model(config,category_ids)
        training_args=SimpleNamespace(learning_rate=config['learning_rate'],temperature=config['temperature'],margin=config['margin'],batch_size=64,training_seed=seed,epochs=30,patience=5,defer_test=True)
        started=time.perf_counter()
        result=train(model,datasets,training_args,torch.device('cpu'),directory/'model.pt',dict(model_kwargs=kwargs,model_type='category_heads' if config.get('category_heads') else 'shared',objective=config['objective'],margin=config['margin'],temperature=config['temperature'],category_ids=category_ids,hidden_dim=256,output_dim=128,config=config,training_seed=seed,candidates_sha256=reference_report['candidates_sha256']))
        result.update(config=config,real_batch_check=check,wall_seconds=time.perf_counter()-started,full_validation=evaluate_full(model,embeddings,context['validation'],examples['validation'],category_ids))
        write_json(directory/'metrics.json',result)
        print('COMPLETED',name,seed,'validation MRR',result['history'][result['best_epoch']-1]['validation']['mrr'],'full',result['full_validation']['mrr'],flush=True)
    if args.stage=='screen':
        summary={}
        for name in CONFIGS:
            r=json.loads((args.output/f'{name}-seed-42/metrics.json').read_text())
            summary[name]={'best_validation_mrr':r['history'][r['best_epoch']-1]['validation']['mrr'],'full_validation_mrr':r['full_validation']['mrr'],'best_epoch':r['best_epoch'],'epochs_run':len(r['history']),'seconds':r['wall_seconds']}
        write_json(args.output/'screen_summary.json',summary)

if __name__=='__main__':
    main()
