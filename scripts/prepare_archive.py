"""Verify and install the separately deposited complete v2.0.0 artifact ZIP."""
import argparse, hashlib, json, os, stat, uuid, zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def safe_members(archive):
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if path.is_absolute() or '..' in path.parts or '\\' in member.filename or ':' in member.filename or stat.S_ISLNK(mode):
            raise ValueError('Unsafe archive member: '+member.filename)
        if not path.parts or path.parts[0] != 'longwall-surrogate-v2.0.0':
            raise ValueError('Unexpected archive layout: '+member.filename)
        yield member

def verify_payload(root):
    checked = 0
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ',1)
        target = (root/relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or sha256(target) != expected:
            raise ValueError('Payload checksum mismatch: '+relative)
        checked += 1
    return checked

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=REPO/'artifacts')
    a = p.parse_args()
    info = json.loads((REPO/'metadata/archive.json').read_text(encoding='utf-8'))
    if not a.archive.is_file() or sha256(a.archive) != info['sha256']:
        p.error('The file does not match the prepared v2.0.0 ZIP SHA256 in metadata/archive.json.')
    area = a.destination.expanduser().resolve()
    area.mkdir(parents=True,exist_ok=True)
    final = area/'longwall-surrogate-v2.0.0'
    if final.exists():
        p.error('Artifact directory already exists. Use it via LONGWALL_SURROGATE_ARTIFACTS or choose another destination.')
    incoming = area/('_incoming_v2_'+uuid.uuid4().hex)
    incoming.mkdir()
    with zipfile.ZipFile(a.archive) as archive:
        members = list(safe_members(archive))
        archive.extractall(incoming,members=members)
    extracted = incoming/'longwall-surrogate-v2.0.0'
    checked = verify_payload(extracted)
    if not extracted.resolve().is_relative_to(area) or not final.resolve().is_relative_to(area):
        raise ValueError('Artifact move must stay within the selected destination.')
    extracted.rename(final)
    incoming.rmdir()  # Empty directory only.
    print(json.dumps({'status':'installed','artifact_root':str(final),'verified_payload_files':checked,
        'shell_hint':'Set LONGWALL_SURROGATE_ARTIFACTS to this path if using a custom destination.'},indent=2))

if __name__ == '__main__':
    main()
