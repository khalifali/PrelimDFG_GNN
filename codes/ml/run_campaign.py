"""One entry point from a bonded campaign directory or extracted artifact."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot',choices=['initial','final'],default='final')
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=50)
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--q',type=int,nargs='+',default=[1,2,3,4])
    p.add_argument('--mode',choices=['grouped','method','size500'],default='grouped')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--verify-splits',type=Path)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    candidates=([a.input/'campaign.json'] if (a.input/'campaign.json').exists() else list(a.input.rglob('campaign.json')))
    candidates=[p for p in candidates if list(p.parent.glob('*/assessment.json'))]
    if len(candidates)!=1:
        raise SystemExit(f'Expected one completed bonded campaign, found {len(candidates)}')
    campaign=candidates[0].parent
    a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((campaign/'campaign.json').read_text())
    (a.output/'source_campaign.json').write_text(json.dumps(source,indent=2))
    here=Path(__file__).resolve().parent
    dataset=a.output/f'{a.snapshot}_dataset.json'
    subprocess.run([sys.executable,str(here/'dataset.py'),str(campaign),'--output',str(dataset),'--snapshot',a.snapshot],check=True)
    command=[sys.executable,str(here/'train.py'),str(dataset),'--output',str(a.output/'training'),
             '--epochs',str(a.epochs),'--patience',str(a.patience),'--mode',a.mode,'--threads',str(a.threads),
             '--seeds',*map(str,a.seeds),'--q',*map(str,a.q)]
    if a.resume:command+=['--resume']
    if a.verify_splits:command+=['--verify-splits',str(a.verify_splits)]
    subprocess.run(command,check=True)
