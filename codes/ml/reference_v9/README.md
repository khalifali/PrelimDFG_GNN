# Authoritative ML reference supplied by the user

These five files are unchanged copies of the latest uploaded ML and preprocessing
sources. `source_manifest.json` records upload names and SHA-256 hashes. They
supersede conflicting earlier uploaded ML versions.

The active entry points are one directory above. The reference training model's
forward computation is tested for exact agreement with the active GNN. These
files are retained for comparison, not executed by the production workflow.

**The bonded DEM model remains authoritative.** The uploaded legacy LIGGGHTS
input files, execution scripts and converters are not part of this directory or
the active workflow. No generation scripts were included in this upload.

The reference workflow documentation refers to v8 and four targets; the actual
v9 source selects mean geometric exposure only. The active pipeline follows the
v9 executable source and uses one final bonded snapshot per agglomerate.
