"""Plot measured shape diversity and orthographic sphere projections."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('campaign', type=Path)
args = p.parse_args()
with (args.campaign/'manifest.csv').open() as f:
    rows = list(csv.DictReader(f))
families = list(dict.fromkeys(r['family'] for r in rows))
selected = [next((r for r in rows if r['family']==fam and int(r['n'])==100),
                 next(r for r in rows if r['family']==fam)) for fam in families]
clouds = [np.loadtxt(args.campaign/r['case']/'particles.csv', delimiter=',', skiprows=1) for r in selected]
clouds = [a/a[0,3] for a in clouds]
limit = max(abs(a[:,:2]).max()+1 for a in clouds)*1.05
fig, axes = plt.subplots(1,len(families),figsize=(3.4*len(families),4), squeeze=False)
for ax, a, r in zip(axes[0],clouds,selected):
    for x,y,z,rad in a[np.argsort(a[:,2])]:
        ax.add_patch(Circle((x,y),rad,facecolor='#4387b8',edgecolor='#173d58',linewidth=.45))
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',title=f"{r['family']}\nN={r['n']}")
    ax.set_axis_off()
fig.suptitle('Representative aggregates | equal scale | orthographic projection')
fig.tight_layout(rect=(0,0,1,.9))
fig.savefig(args.campaign/'representatives.png',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for fam in families:
    subset=[r for r in rows if r['family']==fam]
    for ax,key,title in zip(axes,['rg_over_radius','anisotropy'],['Radius of gyration / particle radius','Shape anisotropy']):
        ax.scatter([int(r['n']) for r in subset],[float(r[key]) for r in subset],alpha=.7,label=fam)
        ax.set(xlabel='Number of particles',ylabel=title)
        ax.grid(alpha=.2)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.campaign/'shape_summary.png',dpi=180)
