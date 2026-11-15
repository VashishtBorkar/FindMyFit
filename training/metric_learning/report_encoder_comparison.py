"""Compare frozen encoders on identical candidates and full held-out category pools."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.models.metric import FashionCompatibilityModel
from findmyfit.models.two_tower import TwoTowerCompatibilityModel
from findmyfit.storage.local_images import LocalImageStore
from training.metric_learning.compare_retrieval import evaluate, write_json
from training.metric_learning.data import load_embeddings
from training.metric_learning.ranking_data import RankingDataset, make_ranking_splits

NAMES = ('shared_contrastive', 'shared_projector', 'two_tower')
SEEDS = (42, 43, 44)


@torch.no_grad()
def full_pool(model, embeddings, context, examples, category_ids):
    if model is not None:
        model.eval()
    result = {}
    for category in sorted({row[1] for row in examples}):
        pool = context['pools'][category]
        anchors = sorted({row[0] for row in examples if row[1] == category})
        index = {item: i for i, item in enumerate(pool)}
        cid = category_ids[category]
        def encode(ids, query):
            chunks = []
            for start in range(0, len(ids), 256):
                x = torch.from_numpy(np.stack([embeddings[i]['embedding'] for i in ids[start:start+256]]))
                if isinstance(model, TwoTowerCompatibilityModel):
                    categories = torch.full((len(x),), cid, dtype=torch.long)
                    x = model.encode_queries(x, categories) if query else model.encode_items(x, categories)
                elif model is not None:
                    x = model(x)
                else:
                    x = torch.nn.functional.normalize(x, dim=-1)
                chunks.append(x)
            return torch.cat(chunks)
        items, queries = encode(pool, False), encode(anchors, True)
        rows = []
        for start in range(0, len(anchors), 64):
            order = (queries[start:start+64] @ items.T).argsort(dim=1, descending=True, stable=True)
            for anchor, ordered in zip(anchors[start:start+64], order):
                relevant = {index[i] for i in context['partners'][anchor] if i in index}
                ranks = [j+1 for j, i in enumerate(ordered.tolist()) if i in relevant]
                assert ranks
                hits = sum(r <= 5 for r in ranks)
                dcg = sum(1 / np.log2(r+1) for r in ranks if r <= 5)
                ideal = sum(1 / np.log2(r+1) for r in range(1, min(5, len(ranks))+1))
                rows.append(dict(mrr=1/min(ranks), hit_at_5=float(min(ranks)<=5),
                                 recall_at_5=hits/len(ranks), ndcg_at_5=dcg/ideal))
        result[category] = dict(queries=len(rows), catalog_items=len(pool),
                               **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
    total = sum(r['queries'] for r in result.values())
    return dict(queries=total, by_category=result,
                **{k: sum(r[k]*r['queries'] for r in result.values())/total
                   for k in ('mrr', 'hit_at_5', 'recall_at_5', 'ndcg_at_5')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    settings = Settings.from_env()
    roots = {'clip': args.runs.parent, 'fashionsiglip': args.runs}
    reports = {encoder: {seed: json.loads((root / (f'controlled-seed-{seed}' if encoder == 'clip' else f'seed-{seed}') / 'metrics.json').read_text())
                         for seed in SEEDS} for encoder, root in roots.items()}
    validation = {encoder: {name: statistics.mean(r['models'][name]['history'][r['models'][name]['best_epoch']-1]['validation']['mrr']
                                                for r in runs.values()) for name in NAMES} for encoder, runs in reports.items()}
    selection = {'mean_validation_mrr': validation,
                 'preferred_model_per_encoder': {e: max(scores, key=scores.get) for e, scores in validation.items()},
                 'test_used_for_selection': False, 'historical_test_previously_examined': True}
    write_json(args.runs / 'selection.json', selection)
    print('VALIDATION SELECTION', selection, flush=True)
    reference = json.loads((roots['clip'] / 'controlled-seed-42/candidates.json').read_text())
    repo = SqliteEmbeddingRepository(create_session_factory(settings.database_url), LocalImageStore(settings.images_dir))
    records = {}
    for encoder, runs in reports.items():
        first = runs[42]
        embeddings, _ = load_embeddings(repo, model_name=first['encoder'][0], model_version=first['encoder'][1])
        digest = hashlib.sha256()
        for item, row in sorted(embeddings.items()):
            digest.update(item.encode()+b'\0'); digest.update(row['embedding'].tobytes())
        assert all(r['vector_sha256'] == digest.hexdigest() for r in runs.values())
        context = {}
        examples, audit = make_ranking_splits(settings.project_root/'data', embeddings, seed=42,
                                             candidate_count=32, anchor_category='top', target_categories=['pants','shoes'], sampling_context=context)
        assert json.loads(json.dumps(examples)) == reference
        category_ids = first['category_ids']
        loaders = {split: DataLoader(RankingDataset(rows, embeddings, category_ids), batch_size=64)
                   for split, rows in examples.items() if split != 'train'}
        raw = {split: dict(sampled=evaluate(None, loader, torch.device('cpu')),
                           full=full_pool(None, embeddings, context[split], examples[split], category_ids))
               for split, loader in loaders.items()}
        records[encoder] = dict(encoder=first['encoder'], vector_sha256=digest.hexdigest(), raw=raw, models={})
        for name in NAMES:
            records[encoder]['models'][name] = {}
            for seed in SEEDS:
                directory = roots[encoder] / (f'controlled-seed-{seed}' if encoder == 'clip' else f'seed-{seed}')
                report = runs[seed]
                assert json.loads((directory/'candidates.json').read_text()) == reference
                assert report['category_ids'] == category_ids
                cp = torch.load(directory/f'{name}.pt', map_location='cpu', weights_only=True)
                model_type = TwoTowerCompatibilityModel if name == 'two_tower' else FashionCompatibilityModel
                model = model_type(**cp['model_kwargs'])
                model.load_state_dict(cp['model_state_dict']); model.eval()
                score = {split: dict(sampled=evaluate(model, loader, torch.device('cpu')),
                                     full=full_pool(model, embeddings, context[split], examples[split], category_ids))
                         for split, loader in loaders.items()}
                expected = report['models'][name]
                assert abs(score['validation']['sampled']['mrr'] - expected['history'][expected['best_epoch']-1]['validation']['mrr']) < 1e-7
                if expected['test'] is not None:
                    assert abs(score['test']['sampled']['mrr'] - expected['test']['mrr']) < 1e-7
                score.update(best_epoch=expected['best_epoch'], epochs_run=len(expected['history']),
                             parameters=expected['parameters'], checkpoint=str(directory/f'{name}.pt'))
                records[encoder]['models'][name][seed] = score
                print('SCORED', encoder, name, seed, score['test']['sampled']['mrr'], flush=True)
                write_json(args.runs/'evaluation.json', records)
        del embeddings
    summary = {}
    for encoder, record in records.items():
        summary[encoder] = {'raw': record['raw'], 'models': {}}
        for name, seeds in record['models'].items():
            aggregate = {}
            for split in ('validation', 'test'):
                aggregate[split] = {}
                for scope in ('sampled', 'full'):
                    keys = ('mrr','recall_at_5','recall_at_10') if scope == 'sampled' else ('mrr','hit_at_5','recall_at_5','ndcg_at_5')
                    aggregate[split][scope] = {k: {'mean': statistics.mean(r[split][scope][k] for r in seeds.values()),
                                                        'sd': statistics.stdev(r[split][scope][k] for r in seeds.values())} for k in keys}
            aggregate['parameters'] = seeds[42]['parameters']
            summary[encoder]['models'][name] = aggregate
    write_json(args.runs/'summary.json', dict(selection=selection, encoders=summary))
    with (args.runs/'comparison.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['encoder','model','seed','best_epoch','parameters','test_recall5','test_mrr','full_test_mrr','full_test_recall5'])
        for encoder, record in records.items():
            for name, seeds in record['models'].items():
                for seed, r in seeds.items():
                    writer.writerow([encoder,name,seed,r['best_epoch'],r['parameters'],r['test']['sampled']['recall_at_5'],r['test']['sampled']['mrr'],r['test']['full']['mrr'],r['test']['full']['recall_at_5']])
    print('SAVED', args.runs/'summary.json', flush=True)

if __name__ == '__main__':
    main()
