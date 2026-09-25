"""Fresh, commit-pinned official download for R1.2B provenance repair.

The old cache is never read. A validated partial fresh run can resume through
per-file sidecars; an incomplete or zero-byte path is always re-downloaded.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST = ROOT / "results/statsbomb360/source_manifest.csv"
OUT = ROOT / "results/statsbomb360_r12b"
RAW = Path("/private/tmp/statsbomb360_r12b_fresh_20260924")
MASTER_BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data/"
API = "https://api.github.com/repos/hudl/open-data/commits/master"
MATCH_COUNT = 115
FILE_COUNT = 348


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_commit() -> str:
    lock = RAW / "_official_commit.txt"
    if lock.is_file():
        value = lock.read_text().strip()
        if len(value) == 40 and all(c in "0123456789abcdef" for c in value):
            return value
        raise RuntimeError("Invalid fresh official commit lock")
    req = Request(API, headers={"User-Agent": "PEACE-FC-GK-provenance-audit", "Accept": "application/vnd.github+json"})
    with urlopen(req, timeout=40) as response:
        if response.status != 200:
            raise RuntimeError(f"Official GitHub API status {response.status}")
        commit = json.load(response)["sha"]
    if len(commit) != 40:
        raise RuntimeError("Could not pin official source commit")
    RAW.mkdir(parents=True, exist_ok=True)
    lock.write_text(commit + "\n")
    return commit


def read_paths() -> list[str]:
    with OLD_MANIFEST.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != FILE_COUNT:
        raise RuntimeError(f"Expected 348 required paths, found {len(records)}")
    paths = []
    for r in records:
        url = r["url"]
        if not url.startswith(MASTER_BASE):
            raise RuntimeError(f"Nonofficial URL in path list: {url}")
        paths.append(url.removeprefix(MASTER_BASE))
    if len(set(paths)) != FILE_COUNT:
        raise RuntimeError("Duplicate required source paths")
    if len([p for p in paths if p.startswith("events/")]) != MATCH_COUNT or \
       len([p for p in paths if p.startswith("lineups/")]) != MATCH_COUNT or \
       len([p for p in paths if p.startswith("three-sixty/")]) != MATCH_COUNT:
        raise RuntimeError("Required match source categories are incomplete")
    return paths


def existing_valid(path: Path, metadata: Path, url: str) -> dict | None:
    if not path.is_file() or not metadata.is_file():
        return None
    try:
        row = json.loads(metadata.read_text())
        if row.get("url") != url or row.get("http_status") != 200 or path.stat().st_size <= 0:
            return None
        h = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                h.update(block)
        if h.hexdigest() != row.get("sha256") or path.stat().st_size != row.get("bytes"):
            return None
        with path.open("rb") as file:
            json.load(file)
        return row
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def fetch(relative: str, commit: str) -> dict:
    url = f"https://raw.githubusercontent.com/hudl/open-data/{commit}/data/{relative}"
    path = RAW / relative
    sidecar = RAW / "_metadata" / (relative + ".verified.json")
    legacy_sidecar = Path(str(path) + ".verified.json")
    if legacy_sidecar.is_file() and not sidecar.is_file():
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy_sidecar, sidecar)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    prior = existing_valid(path, sidecar, url)
    if prior:
        return prior
    last_error = None
    for attempt in range(1, 5):
        partial = Path(str(path) + ".part")
        try:
            req = Request(url, headers={"User-Agent": "PEACE-FC-GK-provenance-audit"})
            sha = hashlib.sha256()
            byte_count = 0
            with urlopen(req, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                with partial.open("wb") as file:
                    while block := response.read(1024*1024):
                        file.write(block)
                        sha.update(block)
                        byte_count += len(block)
            if byte_count <= 0:
                raise RuntimeError("Zero-byte official response")
            with partial.open("rb") as file:
                json.load(file)
            os.replace(partial, path)
            record = {
                "relative_path": relative, "url": url, "downloaded_at_utc": utc_now(),
                "http_status": 200, "bytes": byte_count, "sha256": sha.hexdigest(),
                "json_valid": True, "official_commit": commit,
            }
            sidecar.write_text(json.dumps(record, sort_keys=True) + "\n")
            return record
        except Exception as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Fresh official download failed for {relative}: {last_error}")


def validate_population(paths: list[str]) -> None:
    registry = json.loads((RAW/"competitions.json").read_text())
    selected = [r for r in registry if (r["competition_id"], r["season_id"]) in {(43,106),(55,282)}]
    if len(selected) != 2 or any(r.get("match_available_360") is None for r in selected):
        raise RuntimeError("Competition/season IDs or 360 availability changed")
    match_ids = []
    fresh_match_rows = []
    for comp, season, label in ((43,106,"WC2022"),(55,282,"EURO2024")):
        matches = json.loads((RAW/f"matches/{comp}/{season}.json").read_text())
        selected = [m for m in matches if m.get("match_available_360") is not None or m.get("match_status_360") == "available"]
        available = [m["match_id"] for m in selected]
        match_ids.extend(available)
        fresh_match_rows.extend((label,m) for m in selected)
    if len(match_ids) != MATCH_COUNT:
        raise RuntimeError(f"Official available match count changed: {len(match_ids)}")
    for category in ("events", "lineups", "three-sixty"):
        actual = {int(Path(p).stem) for p in paths if p.startswith(category + "/")}
        if actual != set(match_ids):
            raise RuntimeError(f"Official {category} match IDs differ from frozen population")
    (RAW/"selected_matches.json").write_text(json.dumps(fresh_match_rows, indent=2))


def main() -> None:
    paths = read_paths()
    commit = get_commit()
    print(f"Pinned Hudl open-data commit {commit}; fresh source directory {RAW}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, path, commit): path for path in paths}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(paths):
                print(f"Verified {len(rows)}/{len(paths)} official JSON files", flush=True)
    rows.sort(key=lambda r: r["relative_path"])
    if len(rows) != FILE_COUNT or any(r["bytes"] <= 0 or not r["json_valid"] for r in rows):
        raise RuntimeError("Fresh source verification incomplete")
    validate_population(paths)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"source_manifest_verified.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Clean official source manifest: {len(rows)} files, zero zero-byte entries", flush=True)


if __name__ == "__main__":
    main()
