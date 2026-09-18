"""Compare matched ML reports and geometry without inferring causation from R² alone."""
import sys,csv
from pathlib import Path
import numpy as np
bpm,vdw=map(Path,sys.argv[1:])
lines=['# Matched BPM versus vdW comparison','',
 'Same initial geometries, initial velocities/spins, common material properties, numerical drag, duration, 2048-ray target definition, case exclusions, v9 splits and ML settings.',
 'The mechanical model changes; target distributions may therefore change. Compare RMSE/MAE as well as R². These are fixed-time outputs, not certified equilibria.','']
for name in ('core134_v9','extended179_v9'):
 lines+=['## '+name,'']
 for label,root in [('BPM',bpm),('vdW',vdw)]:
  lines+=['### '+label,'',(root/name/'results.md').read_text(),'']
rows=list(csv.DictReader((vdw/'campaign/paired_geometry.csv').open()))
last={}
for r in rows:
 key=(r['case'],r['model'])
 if key not in last or int(r['step'])>int(last[key]['step']):last[key]=r
lines+=['## Structural changes','', '| Quantity | Median absolute BPM–vdW difference | Maximum |','|---|---:|---:|']
for col in ['contacts','mean_coordination','rg_over_d','exposure','aligned_rms_over_d']:
 delta=[abs(float(r[col])-float(last[(case,'bpm')][col])) for (case,model),r in last.items() if model=='vdw']
 lines.append(f'| {col} | {np.median(delta):.6g} | {max(delta):.6g} |')
lines+=['','For CFD–DEM, identical initial states do not guarantee identical relaxed states. BPM reference bonds constrain relative motion; vdW permits rearrangement and new contacts. Any force-law calibration to match a reference geometry must be independently justified, not selected to recover an ML score.']
(vdw/'bpm_vdw_comparison.md').write_text('\n'.join(lines)+'\n')
