# Public Release Checklist

## Completed
- [x] Repository is public.
- [x] Default branch is `main`.
- [x] MIT license is present for project software/code.
- [x] StatsBomb data source and provider-term distinction are documented.
- [x] AI-assisted research disclosure is public.
- [x] Frozen paper/abstract/supplement PDFs are included.
- [x] Canonical source/model lineage is documented.
- [x] Final event-only winner-lock verifier is included.
- [x] Final StatsBomb 360 R1.2B attribution/provenance scripts and verifier are included.
- [x] No raw StatsBomb JSON is bundled.
- [x] README points to the public repository URL.
- [x] Copy/paste reproduction commands are documented.
- [x] Final pip requirements include statsmodels.

## Final local maintenance step
After pulling the documentation updates in this commit, regenerate:

- `CONTENTS_MANIFEST.md`
- `SHA256SUMS.txt`

from the current repository working tree, commit them, and push.

Do not include `.git/` in either inventory.
