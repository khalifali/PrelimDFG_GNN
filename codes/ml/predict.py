"""Use a saved neural fold model; these predictions are not new test scores."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from dataset import read_dataset, DESCRIPTORS
from models import make_model, predict


def load_checkpoint(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['schema']!=1 or checkpoint['descriptors']!=DESCRIPTORS:
        raise ValueError('Incompatible checkpoint')
    model=make_model(checkpoint['kind'],checkpoint['q'])
    model.load_state_dict(checkpoint['state_dict'])
    scale={k:np.asarray(v) for k,v in checkpoint['scale'].items()}
    return model,scale,checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); model,scale,checkpoint=load_checkpoint(a.checkpoint)
    records=read_dataset(a.dataset);pred,_=predict(model,records,scale,checkpoint['kind'])
    with a.output.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','prediction','seen_during_training'])
        writer.writerows((g['case'],float(v),g['case'] in checkpoint['training_cases']) for g,v in zip(records,pred))
