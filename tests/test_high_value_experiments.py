"""Controls for resampling and target-category heads."""
import unittest
import numpy as np
import torch
from findmyfit.models.category_heads import CategoryHeadCompatibilityModel
from training.metric_learning.ranking_data import ResampledRankingDataset
from training.metric_learning.high_value_experiments import evaluate_full

class HighValueTests(unittest.TestCase):
    def test_resampling_preserves_positives_exclusions_and_seed(self):
        embeddings={str(i):{'embedding':np.ones(4,dtype=np.float32)} for i in range(30)}
        examples=[('0','pants',['1','3','4'],0)]
        context={'partners':{'0':{'1','2'}},'pools':{'pants':[str(i) for i in range(1,30)]}}
        a=ResampledRankingDataset(examples,embeddings,{'pants':0},context,42)
        b=ResampledRankingDataset(examples,embeddings,{'pants':0},context,42)
        a.set_epoch(1);b.set_epoch(1);self.assertEqual(a.examples,b.examples)
        first=a.examples;a.set_epoch(2);self.assertNotEqual(first,a.examples)
        self.assertEqual(a.examples[0][2][0],'1');self.assertNotIn('2',a.examples[0][2])
        self.assertEqual(len(set(a.examples[0][2])),3)
        self.assertEqual(examples[0][2],['1','3','4'])

    def test_heads_route_categories_and_backpropagate(self):
        model=CategoryHeadCompatibilityModel(embedding_dim=4,hidden_dim=8,output_dim=3,category_ids=[0,1])
        x=torch.randn(4,4);c=torch.tensor([0,1,0,1]);output=model.encode(x,c)
        torch.testing.assert_close(output.norm(dim=1),torch.ones(4))
        for i in range(4):
            expected=torch.nn.functional.normalize(model.heads[str(c[i].item())](model.backbone(x[i:i+1])),dim=1)
            torch.testing.assert_close(output[i:i+1],expected)
        model(x,c,torch.randn(4,3,4)).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_full_pool_credits_multiple_positives(self):
        model=torch.nn.Identity()
        embeddings={'a':{'embedding':np.array([1.,0.],dtype=np.float32)}}
        pool=['n0','p','n1','n2','n3','q']
        for item,x in zip(pool,[.99,.9,.8,.7,.6,.5]):
            embeddings[item]={'embedding':np.array([x,(1-x*x)**.5],dtype=np.float32)}
        context={'pools':{'pants':pool},'partners':{'a':{'p','q'}}}
        examples=[('a','pants',['p','n0'],0),('a','pants',['q','n0'],0)]
        result=evaluate_full(model,embeddings,context,examples,{'pants':0})
        self.assertEqual(result['mrr'],.5)
        self.assertEqual(result['recall_at_5'],.5)
        self.assertEqual(result['hit_at_5'],1)
        self.assertEqual(result['queries'],1)
