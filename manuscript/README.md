# Current manuscript: 180-agglomerate study

`agglomerate_exposure.tex` is the working manuscript; `agglomerate_exposure.pdf`
is its compiled review copy. This is the current writing target for the new
180-case study. The earlier 135-case report in `prelim_hbr/` is historical and
its training scores must not be transferred to the new dataset.

Compile using the separate BibTeX database:

```bash
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error agglomerate_exposure.tex
bibtex agglomerate_exposure
pdflatex -interaction=nonstopmode -halt-on-error agglomerate_exposure.tex
pdflatex -interaction=nonstopmode -halt-on-error agglomerate_exposure.tex
```

Editable TikZ diagrams and snapshot placeholders are in `figures/`.
Particle/contact, database and software tables are in `tables/`.
Detailed DEM methods and mechanical parameter tables are in Appendix A.
All internal manuscript notes and placeholders are red.
`references_agglomerate_exposure.bib` retains the old reference entries and adds
Fibonacci sampling and HEALPix sources; only relevant cited entries are printed.
`briesen_comments.md` maps all 26 original PDF annotations to revisions and
explicitly outstanding numerical tests.

The abstract and introduction state the research question. Standard methods,
results, discussion and conclusion sections distinguish completed numerical
work from planned ML analysis. Red bracketed italic text marks incomplete material.
Explain ML terms at first use for readers with an engineering background.

Numerical results currently come from `results/study180/dem_summary.csv`,
`results/study180/exposure_summary_post.json` and
`docs/exposure_refinement.json`. No new ML accuracy is claimed.

Next analysis: identical held-out splits for GNN, linear regression, random
forest and ANN using the six existing conventional descriptors. Recover or
reconstruct the training script, verify descriptor and contact definitions,
freeze splits and training-side model selection, then replace placeholders with
computed scores and figures. Keep reports and slides outside `codes/`.

## Exposure and shielding: literature for the introduction

Use [the annotated literature review](../docs/exposure_shielding_literature.md) when motivating the exposure target. It includes reusable introduction wording, drag/mobility and breakup references, and limits on physical interpretation. Breakage is application context only; the present paper remains separate from the ongoing DFG deagglomeration work.
