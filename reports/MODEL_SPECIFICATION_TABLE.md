# Publication-Grade Model Specification

The machine-readable specification is `results/final_winner_lock/model_specification_table.csv`.

| component | estimand | target | model family | split strategy | primary role |
|---|---|---|---|---|---|
| TF xR | retention probability | R1 | canonical selected family | outer match folds with fit/calibration/evaluation split | primary ex-ante criterion |
| TA-GEO xR | retention conditional on intended geometry | R1 | accepted target-aware family | target-identifiable selected sample | secondary selected-sample analysis |
| V surface | origin possession value | continuation/next-shot value | canonical surface | training matches only | value component |
| success payoff | value after retained action | downstream state value | ShrunkBranchMean | training matches only | success branch |
| failure consequence | value after lost action | failure branch value | ShrunkBranchMean | training matches only | failure branch |
| canonical xT-GK score | expected downstream value under asymmetric loss | rho*S - V - (1-rho)*C | canonical composition | strict OOF | primary value formulation |
