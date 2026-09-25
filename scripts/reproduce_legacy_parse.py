"""Run original parser with only filesystem layout repaired, preserving original code."""
from pathlib import Path
import subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
stage=ROOT/'data/interim/legacy_parse';(stage/'data').mkdir(parents=True,exist_ok=True)
for p in (ROOT/'data/manifests').glob('matches_*.json'):shutil.copy2(p,stage/'data'/p.name)
events=stage/'data/events'
if not events.exists():events.symlink_to(ROOT/'data/raw/events',target_is_directory=True)
command=[sys.executable,str(ROOT/'historical/goalkeeper-in-possession-repo/pipeline.py')]
r=subprocess.run(command,cwd=stage,capture_output=True,text=True)
(ROOT/'results/v2_0_reproduction/path_repaired_parse.txt').write_text('Command: '+repr(command)+'\nCWD: '+str(stage)+'\nModification: input layout only; original script unmodified.\nExit: '+str(r.returncode)+'\n'+r.stdout+r.stderr)
print(r.stdout,r.stderr);sys.exit(r.returncode)
