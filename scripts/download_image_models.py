"""Download and verify the dedicated Ref2VA partition and four source images."""
import argparse
import hashlib
import json
from urllib.request import urlopen
from common import ROOT, MODEL_ROOT
from image_common import REF_MODEL_ROOT, REF_ADAPTER_PATH
from download_models import digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'configs/image_weights.json').read_text())
    adapter = manifest['adapter']
    if not args.verify_only:
        from huggingface_hub import snapshot_download, hf_hub_download
        REF_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        for name in ('model_index.json','modular_model_index.json','audio_scheduler','audio_vae',
                     'processor','scheduler','text_encoder','tokenizer','vae'):
            source, target = MODEL_ROOT/name, REF_MODEL_ROOT/name
            if not source.exists():
                raise SystemExit('First run scripts/download_models.py: missing '+str(source))
            if not target.exists() and not target.is_symlink():
                target.symlink_to(source)
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise SystemExit('Existing shared component differs: '+str(target))
        snapshot_download(manifest['base_repo'], revision=manifest['base_revision'],
                          allow_patterns=[e['name'] for e in manifest['files']],
                          local_dir=REF_MODEL_ROOT, max_workers=2)
        if REF_ADAPTER_PATH.name != adapter['file']:
            raise SystemExit('Keep the original Ref2VA adapter filename')
        hf_hub_download(adapter['repo'], filename=adapter['file'], revision=adapter['revision'],
                        local_dir=REF_ADAPTER_PATH.parent)
        for item in manifest['reference_images']:
            path = ROOT/item['path']
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(item['url'], timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != item['sha256']:
                    raise SystemExit('Reference download hash mismatch')
                path.write_bytes(data)
    for item in manifest['files']:
        path = REF_MODEL_ROOT/item['name']
        assert path.stat().st_size == item['size'], path
        assert digest(path, git_blob=not bool(item['sha256'])) == (item['sha256'] or item['blob']), path
    assert REF_ADAPTER_PATH.stat().st_size == adapter['size']
    assert digest(REF_ADAPTER_PATH) == adapter['sha256']
    for item in manifest['reference_images']:
        assert digest(ROOT/item['path']) == item['sha256'], item['path']
    print('Ref2VA model, adapter and four input images verified.')

if __name__ == '__main__':
    main()
