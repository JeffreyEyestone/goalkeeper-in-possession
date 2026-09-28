# Data package

This directory contains the minimal source-identification and held-out evidence package supporting the SSAC27 submission.

## Event-data study

`event/source_info.json`
: Exact StatsBomb Open Data repository revision and base URL used for the event study.

`event/manifests/`
: Match IDs for Premier League 2015/16, FA WSL 2020/21, World Cup 2018, World Cup 2022, and Euro/Copa America 2024. Together these manifests contain the 722 matches used in the event-data study.

## StatsBomb 360 extension

`360/source_manifest.csv`
: Clean-source provenance manifest for the 348 official StatsBomb JSON files used in the 360 extension. Each row records source path, official URL, byte size, HTTP status, SHA256, JSON validity, and the pinned upstream commit.

`360/nested_oof_predictions.parquet`
: Match-grouped out-of-fold predictions for the nested 360 retention models used in the held-out comparison.

`360/primary_bootstrap.parquet`
: Paired match-cluster bootstrap draws supporting the primary visible-option retention comparison.

## Raw third-party data

Raw StatsBomb event and 360 JSON files are not redistributed in this repository. They remain available from the official provider repository:

https://github.com/hudl/open-data

The included event manifests, pinned event-data revision, and exact 360 source manifest identify the third-party source data used in the research. Publication-level result tables are in `../results/`.

StatsBomb's provider terms govern the underlying provider data.
