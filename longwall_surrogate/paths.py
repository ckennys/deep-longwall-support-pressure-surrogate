"""Resolve complete artifacts without depending on the working directory."""
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def artifact_root():
    override = os.environ.get('LONGWALL_SURROGATE_ARTIFACTS')
    if override:
        return Path(override).expanduser().resolve()
    installed = REPOSITORY_ROOT/'artifacts/longwall-surrogate-v2.0.0'
    if installed.is_dir():
        return installed
    # Also support the complete ZIP layout, with data/models next to the entry scripts.
    return REPOSITORY_ROOT

ROOT = artifact_root()
