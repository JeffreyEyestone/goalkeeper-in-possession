# The Goalkeeper in Possession
## Visible Options, Retention Risk, and Asymmetric Turnover Cost

Minimal public research repository supporting the MIT Sloan Sports Analytics Conference 2027 Soccer-track submission.

### Research question

Goalkeeper distribution is evaluated after the pass is played, but the decision is made earlier: **what options were actually available, how risky was the chosen solution, and what did failure expose?**

The study is organized as:

**OPTIONS → RISK → CONSEQUENCE**

### Headline findings

- **Visible options and choice.** In StatsBomb 360 data, direct play was observed on 85.9% (World Cup 2022) and 91.9% (Euro 2024) of distributions with no clear short exit, versus 27.2% and 25.8% with two or more clear exits.
- **Visible options and retention.** Adding clear-short-option count and a visible central exit to the event-only goalkeeper model improved match-held-out AUC from 0.6485 to 0.6991 (ΔAUC +0.0506; 95% paired match-cluster CI +0.0348 to +0.0660).
- **Environment dependence.** Expected Retention for Goalkeepers (xR-GK) retained moderate discrimination in entirely unseen competitions (AUC 0.645–0.710), while absolute probability calibration shifted by environment.
- **Asymmetric failure cost.** Independently refit cohort models produced a complete lost-branch burden of approximately 2.40–2.90× the value of the possession state surrendered.
- **Target-aware geometry.** Intended-target geometry substantially improved retention discrimination in the target-identifiable sample, but target observation is outcome-selective; this is a supporting result rather than an unbiased all-action ranking model.

The 360 choice relationships are observational and are not interpreted as causal pressing effects.

## Repository contents

### `data/`
Minimal source-identification and held-out evidence package:
- exact event-study match manifests and pinned StatsBomb Open Data revision;
- exact 360 source manifest with URLs, hashes, sizes, validation status, and pinned upstream commit;
- match-held-out 360 model predictions and paired match-cluster bootstrap draws supporting the primary incremental-retention result.

Raw third-party provider JSON is not duplicated here; it remains at the official StatsBomb Open Data repository and is identified exactly by the included manifests and source metadata.

### `results/`
Human-readable publication evidence supporting the reported event-only, target-aware, visible-option, and failure-cost findings.

### `verify_headline_results.py`
A dependency-free verification script for the headline numerical claims stored in the publication evidence files.

Run:

```bash
python3 verify_headline_results.py
```

Expected output:

```text
MINIMAL SLOAN REPO VERIFICATION: PASS
```

## Data source

This research uses **StatsBomb Open Data**, including event-linked StatsBomb 360 freeze frames.

Official provider repository:

https://github.com/hudl/open-data

The event-data cohorts are pinned to the upstream revision recorded in `data/event/source_info.json`. The 360 source files are identified individually in `data/360/source_manifest.csv`.

StatsBomb's provider terms apply to the underlying third-party data. The MIT license in this repository applies only to original project code.

## Scope

This repository intentionally contains only material needed to support the SSAC27 submission and verify its headline numerical results. Development history, internal research-control documents, obsolete papers/figures, coaching-product artifacts, and exploratory intermediate files are not part of this public submission repository.
