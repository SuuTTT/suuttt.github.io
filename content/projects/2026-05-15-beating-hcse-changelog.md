# Changelog: Beating HCSE Project

## [2026-05-15]

### Added
- `/workspace/my_post.md`: Initial Hugo-formatted blog post draft.
- `/workspace/my_post_detailed.md`: Enhanced blog post with formulas, figures, and critical review.
- `/workspace/se-research-projects/beat-hcse/generate_paper_tables.py`: Comprehensive benchmarking script for HSBM/SBM replication.
- `/workspace/se-research-projects/beat-hcse/generate_fast_tables.py`: Optimized benchmarking script for quick replication.
- `/workspace/final_paper_tables.md`: Summary table of replicated results.
- `/workspace/inflection_ours.png`: Comparison chart of entropy drop between HCSE and our method.

### Modified
- `/workspace/se-research-projects/beat-hcse/run_evaluation.py`: 
    - Integrated `multistart_incremental_se_heuristic` and `local_move_incremental`.
    - Implemented **Two-Phase Hybrid Optimization** (Modularity initialization + SE refinement).
    - Fixed HD-SE calculation bug for flat partitions to ensure fair comparison.
- `/workspace/paper.tex`: Copied from LaTeX source zip for analysis.

### Published to GitHub (`suuttt.github.io`)
- `content/projects/2026-05-15-beating-hcse.md`: Finalized blog post.
- `static/images/beat-hcse/stretch_compress.jpg`: Extracted from paper source.
- `static/images/beat-hcse/inflection_points_4.png`: Extracted from paper source.
- `static/images/beat-hcse/inflection_ours.png`: Custom generated comparison chart.
