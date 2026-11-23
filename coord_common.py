"""Read-only shared imports; all writes are confined to this experiment."""
from pathlib import Path
import sys
import json
import hashlib
import importlib.util

HERE = Path(__file__).resolve().parent
LINK = HERE.parent / 'ocr-ner-link-v0'
DATA = HERE.parent / 'ocr-fixtures-v0'
sys.path.insert(0, str(LINK))
from support import digest, sha, read, jsonl, raster, polygon_valid, ASSETS, decode, finalize, ocr_runner
from link import save_masked, bind_ocr, map_candidates
sys.path.insert(0, str(HERE))

IDENTITY_KEYS = ('image_id','ocr_run_id','file_sha256','rgb_sha256','width','height','ocr_sha256')

def identity_of(obj):
    return {k:obj[k] for k in IDENTITY_KEYS}

def confined(path):
    path = Path(path).resolve()
    if path == HERE or not path.is_relative_to(HERE):
        raise ValueError('Output must be within coordinate experiment')
    return path

def write(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def verify_files(files, base=None):
    for name, expected in files.items():
        path = Path(name) if base is None else (Path(base)/name).resolve()
        if base is not None and not path.is_relative_to(Path(base).resolve()):
            raise ValueError('Unconfined sealed file')
        if sha(path) != expected:
            raise ValueError('Hash changed: '+str(path))

def verify_freeze():
    frozen = read(HERE/'METHOD_FREEZE.json')
    if frozen['status'] != 'frozen_after_independent_review':
        raise ValueError('Independent review required')
    verify_files(frozen['files'])
    return frozen
