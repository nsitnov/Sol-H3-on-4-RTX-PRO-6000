"""Download only the pinned T2V components and verify their recorded identities."""
import argparse
import hashlib
import json
from common import ROOT, MODEL_ROOT, ADAPTER_PATH

def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/weights.json').read_text())
    base, adapter = manifest['base'], manifest['adapter']
    if not args.verify_only:
        from huggingface_hub import snapshot_download, hf_hub_download
        snapshot_download(base['repo'], revision=base['revision'],
                          allow_patterns=[e['name'] for e in base['files']],
                          local_dir=MODEL_ROOT, max_workers=2)
        # Preserve the repository's dense-datafree directory layout.
        if ADAPTER_PATH.name != 'adapter_model.safetensors' or ADAPTER_PATH.parent.name != 'dense-datafree':
            raise SystemExit('H3_ADAPTER_PATH must end in dense-datafree/adapter_model.safetensors')
        hf_hub_download(adapter['repo'], filename=adapter['path'], revision=adapter['revision'],
                        local_dir=ADAPTER_PATH.parent.parent)
    for entry in base['files']:
        path = MODEL_ROOT / entry['name']
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise SystemExit(f'Missing or incorrect size: {path}')
        expected = entry['sha256'] or entry['blob']
        if digest(path, git_blob=not bool(entry['sha256'])) != expected:
            raise SystemExit(f'Checksum mismatch: {path}')
    if ADAPTER_PATH.stat().st_size != adapter['bytes'] or digest(ADAPTER_PATH) != adapter['sha256']:
        raise SystemExit('Adapter checksum mismatch')
    print(f'Verified {len(base["files"])} model files and the dense-datafree adapter.')

if __name__ == '__main__':
    main()

