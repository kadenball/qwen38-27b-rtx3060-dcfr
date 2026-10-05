#!/usr/bin/env python3
"""Maintainer helper: refresh the release manifest, excluding ignored artifacts."""
import hashlib
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
names = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
rows = []
for name in sorted(set(names) - {'', 'SHA256SUMS'}):
    path = root / name
    if not path.is_file():
        raise SystemExit(f'Missing release file: {name}')
    rows.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n')
(root / 'SHA256SUMS').write_text(''.join(rows))
print(f'Pinned {len(rows)} release files.')
