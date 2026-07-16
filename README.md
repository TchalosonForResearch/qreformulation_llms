# Reproducibility scripts

The numeric prefixes are intentional. They indicate the execution order within each experimental pipeline and make the workflow easier to reproduce. Active pipeline scripts use the `NN_descriptive_name.py` convention. Utilities and debugging helpers are kept in `scripts/tools/` without a numeric prefix, and superseded code is isolated in `scripts/legacy/`.



## LegalBench-RAG mini pipeline

Run the scripts from the project root in this order:

1. `scripts/legalbench/01_check_legalbench_rag_data.py`
2. `scripts/legalbench/02_inspect_legalbench_snippets.py`
3. `scripts/legalbench/03_prepare_legalbench_rag_mini.py`
4. `scripts/legalbench/04_bm25_legalbench_original.py`
5. `scripts/legalbench/05_bm25_legalbench_original_text_variants.py`
6. `scripts/legalbench/06_select_canonical_bm25_baseline.py`
7. `scripts/legalbench/07_prepare_queries_for_reformulation.py`
8. `scripts/legalbench/08_generate_deepseek_reformulations_api.py`
9. `scripts/legalbench/09_inspect_normalize_legalbench_reformulations.py`
10. `scripts/legalbench/10_bm25_legalbench_reformulations.py`
11. `scripts/legalbench/11_diagnose_anchor_preservation.py`
12. `scripts/legalbench/12_rrf_original_reformulation.py`
13. `scripts/legalbench/13_rrf_harm_rate_and_stats.py`
14. `scripts/legalbench/14_rrf_multiview_and_rcs_simple.py`
15. `scripts/legalbench/15_multiview_harm_rate_and_stats.py`
16. `scripts/legalbench/16_build_legalbench_final_synthesis_table.py`
17. `scripts/legalbench/17_build_cross_dataset_final_synthesis_table.py`
18. `scripts/legalbench/22_legalbench_chunk_only_ablation.py`
19. `scripts/legalbench/23_legalbench_dense_hybrid_baselines.py`

The API script reads the DeepSeek credential from the `DEEPSEEK_API_KEY` environment variable. Do not store credentials in source files.

## BSARD pipeline

1. `scripts/bsard/02_bm25_bsard_original.py`
2. `scripts/bsard/03_evaluate_run.py`
3. `scripts/bsard/04_inspect_existing_reformulations.py`
4. `scripts/bsard/05_normalize_existing_reformulations.py`
5. `scripts/bsard/06_bm25_reformulations.py`
6. `scripts/bsard/07_harm_rate.py`
7. `scripts/bsard/08_summarize_bm25_reformulation_results.py`
8. `scripts/bsard/09_rrf_fusion_original_reformulation.py`
9. `scripts/bsard/11_rrf_fusion_harm_rate.py`
10. `scripts/bsard/12_stats_rrf_fusion.py`
11. `scripts/bsard/13_rrf_fusion_original_all_reformulations.py`
12. `scripts/bsard/14_rrf_all_reformulations_harm_and_stats.py`
13. `scripts/bsard/15_consensus_quadrant_analysis.py`
14. `scripts/bsard/16_rcs_simple.py`
15. `scripts/bsard/17_rcs_simple_harm_and_stats.py`
16. `scripts/bsard/18_rcs_grid_search.py`
17. `scripts/bsard/19_rcs_selected_grid_stats.py`
18. `scripts/bsard/20_consensus_quadrant_query_level_stats.py`
19. `scripts/bsard/21_make_bm25_final_synthesis_table.py`

Scripts 15--20 support the exploratory consensus analyses reported in the supplementary material.

## Canonical statistical and anchor analyses

After the dataset-specific outputs are available:

1. `scripts/analysis/24_inventory_experiment_artifacts.py`
2. `scripts/analysis/26_merge_inferential_statistics_phase1b.py`
3. `scripts/analysis/27_complete_bsard_replacement_inference_phase1c.py`
4. `scripts/analysis/28_fix_phase1_final_status_flags.py`
5. `scripts/analysis/30_anchor_proxy_phase2a.py`
6. `scripts/analysis/31_anchor_proxy_phase2b_refined.py`

## Utilities

- `scripts/tools/debug_gpt_jsonl_errors.py`: inspect malformed GPT JSONL records.
- `scripts/tools/fix_appendix_table_display.py`: post-process the LaTeX appendix table.

## Legacy code

`scripts/legacy/23_legalbench_dense_hybrid_baselines_legacy.py` is the earlier dense/hybrid implementation. The active workflow uses `scripts/legalbench/23_legalbench_dense_hybrid_baselines.py`.
