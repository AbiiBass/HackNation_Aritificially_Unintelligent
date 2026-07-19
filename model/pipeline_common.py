"""
pipeline_common.py
Shared helpers used across the pipeline: encode_joint_class (defined once
here, not duplicated per-script), XGBOOST_DETERMINISTIC_PARAMS (forces
tree_method='exact' since the default 'hist' isn't bit-identical across CPU
architectures), and save_json_with_metadata (stamps reports with timestamp +
library versions).
"""

import json
import platform
import subprocess
from datetime import datetime, timezone

CLASS_NAMES = {0: "neither", 1: "thyroid_only", 2: "pcos_only", 3: "both"}

# Include in every XGBClassifier(...) call to reduce cross-machine drift.
XGBOOST_DETERMINISTIC_PARAMS = {
    "tree_method": "exact",
}


def encode_joint_class(pcos_diagnosis, thyroid_dysfunction):
    """Single definition of the joint_class formula. Works on scalars,
    pandas Series, or numpy arrays."""
    return pcos_diagnosis * 2 + thyroid_dysfunction


def _library_versions() -> dict:
    versions = {}
    for name in ["pandas", "sklearn", "xgboost", "joblib"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _git_commit_if_available() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "not a git repo (or git unavailable) -- consider `git init` for real provenance tracking"


def save_json_with_metadata(data: dict, path: str):
    """Wraps a results dict with a metadata block (timestamp, platform,
    library versions, git commit)."""
    stamped = {
        "_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "library_versions": _library_versions(),
            "git_commit": _git_commit_if_available(),
        },
        **data,
    }
    with open(path, "w") as f:
        json.dump(stamped, f, indent=2)