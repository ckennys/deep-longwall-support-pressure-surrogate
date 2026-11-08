"""Verify the complete artifact ZIP's extracted files using the standard library."""
from longwall_surrogate.paths import ROOT
import hashlib, json

def main():
    manifest = ROOT/'SHA256SUMS.txt'
    if not manifest.is_file():
        raise SystemExit('Install the complete artifact ZIP with scripts/prepare_archive.py first.')
    checked, failures = 0, []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ',1)
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            failures.append(name)
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                h.update(block)
        if h.hexdigest() != expected:
            failures.append(name)
        checked += 1
    print(json.dumps({'checked':checked,'failures':failures},indent=2))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
