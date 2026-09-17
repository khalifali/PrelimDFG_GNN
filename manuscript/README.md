# Current manuscript: 180-agglomerate study

`agglomerate_exposure.tex` is the working manuscript; `agglomerate_exposure.pdf`
is its compiled review copy. This is the current writing target for the new
180-case study. The earlier 135-case report in `prelim_hbr/` is historical and
its training scores must not be transferred to the new dataset.

Compile from the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript manuscript/agglomerate_exposure.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript manuscript/agglomerate_exposure.tex
```

The abstract and introduction state the research question. Standard methods,
results, discussion and conclusion sections distinguish completed numerical
work from planned ML analysis. Bracketed italic text marks incomplete material.
Explain ML terms at first use for readers with an engineering background.

Numerical results currently come from `results/study180/dem_summary.csv`,
`results/study180/exposure_summary_post.json` and
`docs/exposure_refinement.json`. No new ML accuracy is claimed.

Next analysis: identical held-out splits for GNN, linear regression, random
forest and ANN using the six existing conventional descriptors. Recover or
reconstruct the training script, verify descriptor and contact definitions,
freeze splits and training-side model selection, then replace placeholders with
computed scores and figures. Keep reports and slides outside `codes/`.
