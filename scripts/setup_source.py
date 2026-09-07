"""Fetch the pinned upstream into a new directory, then apply the exact patch."""
import subprocess
from common import ROOT, SANA_ROOT, SANA_REVISION

def main():
    if SANA_ROOT.exists():
        raise SystemExit(f'{SANA_ROOT} already exists. Use a fresh SANA_ROOT; no existing checkout is modified.')
    SANA_ROOT.mkdir(parents=True)
    def git(*args):
        return subprocess.run(['git', '-C', str(SANA_ROOT), *args], check=True)
    git('init')
    git('remote', 'add', 'origin', 'https://github.com/NVlabs/Sana.git')
    git('fetch', '--depth=1', 'origin', SANA_REVISION)
    git('checkout', '--detach', 'FETCH_HEAD')
    patch = str(ROOT / 'patches/sol-h3-encoder-tp.patch')
    git('apply', '--check', patch)
    git('apply', patch)
    git('add', '-N', 'models/minimax_h3/Sol-H3/h3_runtime/encoder_tp.py')
    git('diff', '--check')
    print(f'Pinned and patched source ready: {SANA_ROOT}')

if __name__ == '__main__':
    main()
