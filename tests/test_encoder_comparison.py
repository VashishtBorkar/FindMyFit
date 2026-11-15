"""Check multi-positive full-pool evaluation and wider encoder inputs."""
import unittest

import numpy as np
import torch

from findmyfit.models.metric import FashionCompatibilityModel
from findmyfit.models.two_tower import TwoTowerCompatibilityModel
from training.metric_learning.compare_retrieval import check_training_batch
from training.metric_learning.loss import CandidateContrastiveLoss, CandidateRankingLoss
from training.metric_learning.ranking_data import RankingDataset
from training.metric_learning.report_encoder_comparison import full_pool


class EncoderComparisonTests(unittest.TestCase):
    def test_full_pool_credits_multiple_known_partners(self):
        vectors = {'anchor': np.array([1., 0.], dtype=np.float32)}
        for i, cosine in enumerate([1., .8, .6, .4, .2, -1.]):
            vectors[str(i)] = np.array([cosine, np.sqrt(1-cosine**2)], dtype=np.float32)
        embeddings = {i: {'embedding': v} for i, v in vectors.items()}
        context = {'pools': {'pants': [str(i) for i in range(6)]},
                   'partners': {'anchor': {'1', '5'}}}
        result = full_pool(None, embeddings, context,
                           [('anchor', 'pants', ['1', '0'], 0)], {'pants': 0})
        self.assertAlmostEqual(result['mrr'], .5)
        self.assertAlmostEqual(result['hit_at_5'], 1.)
        self.assertAlmostEqual(result['recall_at_5'], .5)
        self.assertEqual(result['by_category']['pants']['catalog_items'], 6)

    def test_siglip_inputs_support_all_training_families(self):
        rng = np.random.default_rng(42)
        embeddings = {str(i): {'embedding': rng.normal(size=768).astype(np.float32)} for i in range(5)}
        dataset = RankingDataset([('0', 'pants', ['1', '2', '3', '4'], 0)], embeddings, {'pants': 0})
        for model, criterion in (
            (FashionCompatibilityModel(768), CandidateContrastiveLoss()),
            (FashionCompatibilityModel(768), CandidateRankingLoss()),
            (TwoTowerCompatibilityModel(768, 1), CandidateRankingLoss()),
        ):
            with self.subTest(model=type(model).__name__, loss=type(criterion).__name__):
                check = check_training_batch(model, dataset, torch.device('cpu'), criterion)
                self.assertTrue(check['finite_gradients'])


if __name__ == '__main__':
    unittest.main()
