# Final manuscript corrections

This package is based on the latest full LaTeX manuscript supplied by the
author. No reported metric, table, experiment, limitation or qualitative
observation was silently removed.

## Corrections applied

1. **Construction validity (Section 12.4)**
   - Reconciled the limitation text with the completed 40-query manual audit.
   - States that the sample covers 20% of LegalBench-RAG mini.
   - States that the audit used a single reviewer and therefore provides no
     inter-annotator agreement statistic.
   - Retains the limits of the conservative proxy and the need for larger or
     multi-annotator validation.

2. **Figure 4**
   - Regenerated with the labels `DeepSeek`, `GPT` and `HyDE`.
   - Numerical values and confidence intervals are unchanged.

3. **Anchor diagnostic table**
   - Updated the selected task-level retention values to the final v3 proxy.

4. **Qualitative taxonomy**
   - Preserved all seven proposed mechanisms.
   - Removed the unfinished frequency-chart placeholder.
   - Reframed the taxonomy as a qualitative coding framework because the full
     harmed set was not independently coded.

5. **Illustrative source-sensitivity cases**
   - Preserved the three original examples and their reformulations.
   - Removed the unfilled `Document retrieved` cells.
   - Recast the table as a mechanism-level illustration, without claiming that
     a specific wrong document was manually verified.

6. **RCS sensitivity**
   - Removed submission placeholders and `High priority` language.
   - Preserved the distinction between the completed BSARD grid and the fixed
     LegalBench diagnostic configurations.
   - Explicitly limits the LegalBench claim: no exhaustive grid,
     hyperparameter-robustness claim or out-of-sample optimisation claim.

7. **Original-query ablation**
   - Clarified that the matched one-view comparison is complete.
   - Preserved the limitation that a full multi-view without-original matrix
     was not evaluated.

8. **Discussion and operational monitoring**
   - Updated the old statement that APR correlation and manual validation were
     still pending.
   - Preserved direct document-level source adjudication as future work.

9. **Consistency cleanup**
   - Removed all `To be produced`, `to be completed`, and equivalent pending
     placeholders.
   - Corrected `Mutual rank` to `Reciprocal rank`.

## Files

- `legal_query_reformulation_corrected_final.tex`: corrected full manuscript.
- `figs/`: all manuscript figures, including the corrected Figure 4.
- `tables/`: all external LaTeX/CSV tables used by the manuscript.

## Compilation

Place your existing `cas-refs.bib` beside the main `.tex` file, then run:

```powershell
latexmk -pdf legal_query_reformulation_corrected_final.tex
```

A structural and layout compilation was successfully completed in a temporary
validation copy with the bibliography call disabled, because `cas-refs.bib`
was not included in the uploaded materials. That validation produced 34 pages
and confirmed that all `\input` and `\includegraphics` targets resolve.
