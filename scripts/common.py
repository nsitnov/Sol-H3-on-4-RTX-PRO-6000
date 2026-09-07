"""Portable paths shared by the replication scripts; no GPU initialization."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANA_REVISION = '2936c47637380842aaa4a4488fac5006cc542b70'
SANA_ROOT = Path(os.environ.get('SANA_ROOT', ROOT / 'vendor/Sana')).resolve()
RUNTIME_ROOT = SANA_ROOT / 'models/minimax_h3/Sol-H3'
MODEL_ROOT = Path(os.environ.get('H3_MODEL_ROOT', ROOT / 'models/MiniMax-H3')).resolve()
ADAPTER_PATH = Path(os.environ.get('H3_ADAPTER_PATH', ROOT / 'models/FastH3/dense-datafree/adapter_model.safetensors')).resolve()
OUT = Path(os.environ.get('H3_RUN_ROOT', ROOT / 'runs')).resolve()
CASES_PATH = ROOT / 'configs/cases.json'
sys.path.insert(0, str(RUNTIME_ROOT))

