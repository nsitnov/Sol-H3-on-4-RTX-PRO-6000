"""Additional paths for the I2V / Ref2VA campaign."""
import os
from pathlib import Path
from common import ROOT

REF_MODEL_ROOT = Path(os.environ.get('H3_REF_MODEL_ROOT', ROOT/'models/MiniMax-H3-Ref2VA')).resolve()
REF_ADAPTER_PATH = Path(os.environ.get('H3_REF_ADAPTER_PATH', ROOT/'models/LightX/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors')).resolve()
IMAGE_CASES_PATH = ROOT/'configs/image_cases.json'
