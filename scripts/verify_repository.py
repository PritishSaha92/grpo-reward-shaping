"""Check the delivered files before reproducing any result (standard library only)."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'checksums.json').read_text()); problems=[]
    for name,expected in manifest.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:problems.append(name)
    if problems:raise SystemExit('Missing/changed files: '+', '.join(problems))
    print(f'PASS: {len(manifest)} release files match checksums.json.')
    print('This validates the public export, not an external preregistration or a new GPU execution.')
if __name__=='__main__':main()
