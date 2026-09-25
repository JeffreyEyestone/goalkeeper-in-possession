# Data sources

## Hudl StatsBomb open data

Official repository:

https://github.com/hudl/open-data

This research uses StatsBomb open event data and event-linked StatsBomb 360 freeze-frame data.

For the final 360 extension, the provenance pipeline pins one upstream Hudl open-data commit and records URL, byte size, JSON validity and SHA256 for every required source file in:

`results/statsbomb360_r12b/source_manifest_verified.csv`

The final clean manifest contains 348 nonzero validated JSON files. A fresh download reproduces the frozen 4,831-distribution / 27,777-candidate option tables without unexplained scientific-field differences.

Raw provider JSON is intentionally not bundled in this repository. It is retrieved from the official source by the reproduction scripts.

## Attribution / provider terms

The official StatsBomb Open Data README states that published/shared analysis based on the data should identify **StatsBomb** as the data source and use the StatsBomb logo available from its Media Pack.

Official terms/source:
https://github.com/hudl/open-data

The MIT license in this repository applies to this project's software/code only. Third-party data remain governed by their source-provider terms.
