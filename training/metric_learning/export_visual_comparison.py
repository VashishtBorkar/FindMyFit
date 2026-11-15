"""Export model predictions with portable image keys for local visual review."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.models.metric import FashionCompatibilityModel
from findmyfit.models.two_tower import TwoTowerCompatibilityModel
from findmyfit.storage.local_images import LocalImageStore
from training.metric_learning.compare_retrieval import write_json
from training.metric_learning.ranking_data import make_ranking_splits


@torch.no_grad()
def export(runs, output, count=50, sample_seed=2026, training_seed=42):
    settings = Settings.from_env()
    torch.set_num_threads(4)
    repo = SqliteEmbeddingRepository(create_session_factory(settings.database_url), LocalImageStore(settings.images_dir))
    reference = runs.parent / f'controlled-seed-{training_seed}'
    reports = {
        'clip': (reference, json.loads((reference/'metrics.json').read_text())),
        'fashionsiglip': (runs/f'seed-{training_seed}', json.loads((runs/f'seed-{training_seed}/metrics.json').read_text())),
    }
    bundle = {'schema_version': 1, 'training_seed': training_seed, 'sample_seed': sample_seed,
              'pool': 'full held-out test category pools', 'items': {}, 'queries': [], 'models': []}
    chosen = None
    for encoder, (directory, report) in reports.items():
        embeddings = {}
        metadata = {}
        digest = hashlib.sha256()
        for record in repo.iter_vectors(model_name=report['encoder'][0], model_version=report['encoder'][1]):
            embeddings[record.item_id] = {'embedding': record.vector.copy(), 'category': record.category, 'image_hash': record.image_hash}
            metadata[record.item_id] = {'image_key': record.image_key, 'category': record.category}
        for item, record in sorted(embeddings.items()):
            digest.update(item.encode()+b'\0'); digest.update(record['embedding'].tobytes())
        if digest.hexdigest() != report['vector_sha256']:
            raise ValueError('Encoder vectors differ from the trained run')
        context = {}
        examples, _ = make_ranking_splits(settings.project_root/'data', embeddings, seed=42, candidate_count=32,
                                         anchor_category='top', target_categories=['pants','shoes'], sampling_context=context)
        if json.loads((directory/'candidates.json').read_text()) != json.loads(json.dumps(examples)):
            raise ValueError('Candidate manifest differs from the trained run')
        test = context['test']
        grouped = {category: {row[0] for row in examples['test'] if row[1] == category} for category in ('pants','shoes')}
        eligible = sorted(grouped['pants'] & grouped['shoes'])
        if chosen is None:
            chosen = random.Random(sample_seed).sample(eligible, min(count, len(eligible)))
            bundle['eligible_anchors'] = len(eligible)
            bundle['queries'] = [{'anchor': anchor, 'partners': {category: sorted(test['partners'][anchor] & set(test['pools'][category])) for category in ('pants','shoes')}, 'predictions': {}} for anchor in chosen]
        elif not set(chosen).issubset(eligible):
            raise ValueError('Encoder item coverage differs')
        for name in ('cosine', 'shared_contrastive', 'shared_projector', 'two_tower'):
            model_id = f'{encoder}/{name}'
            if name == 'cosine':
                model = None
            else:
                cp = torch.load(directory/f'{name}.pt', map_location='cpu', weights_only=True)
                cls = TwoTowerCompatibilityModel if name == 'two_tower' else FashionCompatibilityModel
                model = cls(**cp['model_kwargs']); model.load_state_dict(cp['model_state_dict']); model.eval()
            bundle['models'].append({'id': model_id, 'encoder': encoder, 'method': name,
                                     'encoder_identity': report['encoder'], 'encoder_fingerprint': report['encoder_fingerprint'],
                                     'vector_sha256': report['vector_sha256']})
            for category in ('pants','shoes'):
                pool = test['pools'][category]
                cid = report['category_ids'][category]
                def encode(ids, query):
                    chunks = []
                    for start in range(0,len(ids),256):
                        x = torch.from_numpy(np.stack([embeddings[i]['embedding'] for i in ids[start:start+256]]))
                        if isinstance(model, TwoTowerCompatibilityModel):
                            categories = torch.full((len(x),),cid,dtype=torch.long)
                            x = model.encode_queries(x,categories) if query else model.encode_items(x,categories)
                        elif model is not None:
                            x = model(x)
                        else:
                            x = torch.nn.functional.normalize(x,dim=-1)
                        chunks.append(x)
                    return torch.cat(chunks)
                scores = encode(chosen, True) @ encode(pool, False).T
                order = scores.argsort(dim=1, descending=True, stable=True)[:,:5]
                for query, row_scores, row_order in zip(bundle['queries'], scores, order):
                    query['predictions'].setdefault(model_id,{})[category] = [
                        {'item': pool[i], 'score': float(row_scores[i])} for i in row_order.tolist()]
            print('EXPORTED',model_id,flush=True)
        bundle['pool_sizes'] = {c:len(test['pools'][c]) for c in ('pants','shoes')}
        used = set(chosen)
        for query in bundle['queries']:
            for partners in query['partners'].values(): used.update(partners)
            for predictions in query['predictions'].values():
                for rows in predictions.values(): used.update(row['item'] for row in rows)
        bundle['items'].update({item:metadata[item] for item in used})
        del embeddings
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,bundle)
    print('SAVED',output,'queries',len(chosen),'items',len(bundle['items']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('examples/model_comparison/predictions.json'))
    parser.add_argument('--queries',type=int,default=50)
    args=parser.parse_args()
    if args.queries<1: parser.error('--queries must be positive')
    export(args.runs,args.output,args.queries)

if __name__=='__main__':main()
