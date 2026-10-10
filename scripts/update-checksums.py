#!/usr/bin/env python3
"""Refresh the release manifest. Stage intended new files first; ignore untracked work."""
import hashlib
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
rows = []
for name in sorted(set(names) - {'', 'SHA256SUMS'}):
    path = root / name
    if not path.is_file():
        raise SystemExit(f'Missing release file: {name}')
    rows.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n')
(root / 'SHA256SUMS').write_text(''.join(rows))
print(f'Pinned {len(rows)} release files.')
