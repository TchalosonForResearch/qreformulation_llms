PHASE 2 MANUSCRIPT INTEGRATION
================================

Main file:
  legal_query_reformulation_integrated_phase2.tex

New figure:
  figs/figure4_anchor_preservation_final.pdf

New tables/audits:
  tables/anchor_proxy_pooled_recall100.tex
  tables/anchor_proxy_pooled_clustered_correlations_v3.csv
  tables/anchor_proxy_subgroup_correlations_v3.csv
  tables/anchor_proxy_validation_sample_v3_reviewed.csv
  tables/anchor_proxy_validation_summary_v3.csv
  tables/anchor_proxy_validation_report_v3.txt

Scientific interpretation:
  The measure is a manually audited conservative lexical
  anchor-preservation proxy. It is not a complete APR gold standard.
  The pooled association with Delta Recall@100 is rho=0.179,
  95% cluster-bootstrap CI [0.116, 0.242], p_Holm=0.0012.

Manual audit:
  40 unique queries; 16 exact full-component matches;
  100/149 manual tokens recovered; zero spurious extracted tokens;
  descriptive precision=1.000, recall=0.671, F1=0.803.
