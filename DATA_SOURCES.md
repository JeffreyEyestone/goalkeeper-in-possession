# Data sources

## Hudl StatsBomb open data
Official repository: https://github.com/hudl/open-data

The research uses StatsBomb open event data and event-linked 360 freeze-frame data. For the final 360 extension, the provenance pipeline pins an upstream Hudl open-data commit and records URL, byte size, JSON validity and SHA256 for every required source file in:

`results/statsbomb360_r12b/source_manifest_verified.csv`

The final clean manifest contains 348 nonzero validated JSON files. A fresh download reproduces the frozen 4,831-distribution / 27,777-candidate option tables without unexplained scientific-field differences.

Raw provider data remain subject to source-provider terms and are retrieved from the official source rather than bundled into this repository.
