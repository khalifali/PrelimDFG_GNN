"""Scientific-integrity checks for feature construction, grouping and inference."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from dataset import graph_features, read_dataset
from models import GNN, batch_graphs, scaler, predict
from predict import load_checkpoint
from train import splits, save_checkpoint, train_neural


def synthetic():
    result=[]
    for group in range(20):
        for replicate in range(2):
            p=np.column_stack([np.arange(4),np.zeros((4,2))])
            g=graph_features(p,np.full(4,.5))
            g.update(case=f'{group}_{replicate}',group=str(group),method=['a','b','c'][group%3],n=500 if group>=16 else 4,target=.4+group*.01)
            result.append(g)
    return result


class PipelineTests(unittest.TestCase):
    def test_contact_definition_and_descriptors(self):
        g=graph_features([[0,0,0],[1,0,0],[2,0,0]],np.full(3,.5))
        np.testing.assert_allclose(g['descriptors'],[3,np.sqrt(2/3),2,2/3,4/3,2])
        np.testing.assert_array_equal(g['nodes'][:,1],[1,2,1])
        self.assertEqual(g['edge'].shape,(2,4))
        h=graph_features([[0,0,0],[1.000002,0,0]],np.full(2,.5))
        self.assertEqual(h['edge'].shape,(2,0))

    def test_scaling_translation_and_units(self):
        xyz=np.array([[0.,0,0],[1,0,0],[1,1,0]])
        a=graph_features(xyz,np.full(3,.5));b=graph_features((xyz+7)*1e-6,np.full(3,.5e-6))
        for k in a:
            np.testing.assert_allclose(a[k],b[k],atol=1e-6)

    def test_group_leakage(self):
        records=synthetic()
        for mode in ('grouped','method','size500'):
            seen=[]
            for outer,test,fit,val in splits(records,mode):
                group=lambda idx:{records[i]['group'] for i in idx}
                self.assertFalse(group(outer)&group(test));self.assertFalse(group(fit)&group(val))
                self.assertEqual(set(outer),set(fit)|set(val));seen.extend(test)
            self.assertEqual(len(seen),len(set(seen)))
            if mode!='size500':self.assertEqual(len(seen),len(records))

    def test_gnn_permutation_and_batch_independence(self):
        torch.manual_seed(7);torch.set_num_threads(1)
        xyz=np.array([[0.,0,0],[1,0,0],[1,1,0],[2,1,0]])
        a=graph_features(xyz,np.full(4,.5));b=graph_features(xyz[[2,0,3,1]],np.full(4,.5))
        model=GNN().eval()
        with torch.no_grad():
            single=model(batch_graphs([a],np.zeros(2),np.ones(2)))
            pair=model(batch_graphs([a,b],np.zeros(2),np.ones(2)))
        torch.testing.assert_close(pair,single.expand(2),atol=1e-5,rtol=1e-5)

    def test_matches_uploaded_v9_forward(self):
        import importlib.util
        from torch_geometric.data import Data
        spec=importlib.util.spec_from_file_location('reference_v9',Path(__file__).parent/'reference_v9/train_gnn_pure_graph_latent_v9.py')
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        torch.manual_seed(7)
        original=reference.Model(nin=2,ein=1,nout=1,latent=2).eval()
        model=GNN(q=2).eval();model.load_state_dict(original.state_dict())
        batch=batch_graphs(synthetic()[:2],np.zeros(2),np.ones(2))
        nodes,pos,edges,assignment=batch
        src,dst=edges
        data=Data(x=nodes,pos=pos,edge_index=edges,edge_attr=(pos[dst]-pos[src]).norm(dim=1,keepdim=True),batch=assignment)
        with torch.no_grad():
            old,old_q=original(data);new,new_q=model(batch,return_latent=True)
        torch.testing.assert_close(new,old.flatten(),rtol=0,atol=0)
        torch.testing.assert_close(new_q,old_q,rtol=0,atol=0)

    def test_checkpoint_and_training_only_scaler(self):
        data=synthetic();train=data[:12];held=data[12:]
        scale=scaler(train)
        self.assertAlmostEqual(scale['target_mean'],np.mean([g['target'] for g in train]))
        model,scale,epochs,_=train_neural(train,held,'ann',1,7,2,2)
        before,_=predict(model,held,scale,'ann')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';save_checkpoint(path,model,scale,'ann',1,epochs,train)
            restored,s,c=load_checkpoint(path);after,_=predict(restored,held,s,'ann')
            np.testing.assert_allclose(before,after,rtol=0,atol=0)
            self.assertEqual(c['training_cases'],[g['case'] for g in train])

if __name__=='__main__':unittest.main()
