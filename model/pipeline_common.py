"""
pipeline_common.py
Shared logic used by multiple scripts in the pipeline, added post-review
to fix several "defined in two places" bugs found during a full-file audit:

- joint_class encoding was hand-duplicated in label_engineering.py and
  label_noise_sensitivity.py. Now defined ONCE here.
- XGBoost's default tree_method ('hist') isn't guaranteed bit-identical
  across CPU architectures even with a fixed random_state -- this is why
  meta_learner_metrics.json (0.866, built on x86_64 Linux) and a later
  run of label_noise_sensitivity.py (0.878, run on ARM64 Mac) disagreed
  on the exact same test set. XGBOOST_DETERMINISTIC_PARAMS below forces
  tree_method='exact', which IS deterministic given the same data --
  our datasets are small enough (max 3,163 rows) that the speed cost is
  irrelevant. Import and use these params in every XGBClassifier(...) call.
- save_json_with_metadata() stamps every generated report with a
  timestamp + library versions, so staleness (like the old
  robustness_checks.json missing a newly-added check) is detectable
  just by looking at the file, not by memory of which script version
  produced it.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone

CLASS_NAMES = {0: "neither", 1: "thyroid_only", 2: "pcos_only", 3: "both"}

# XGBoost params to include in every XGBClassifier(...) call across the
# project, to minimize (not eliminate -- see requirements.txt) cross-machine
# result drift.
XGBOOST_DETERMINISTIC_PARAMS = {
    "tree_method": "exact",
}


def encode_joint_class(pcos_diagnosis, thyroid_dysfunction):
    """The ONE place this formula is defined. Works on scalars, pandas
    Series, or numpy arrays (relies on * and + broadcasting normally)."""
    return pcos_diagnosis * 2 + thyroid_dysfunction


def decode_joint_class(joint_class):
    """Inverse of encode_joint_class: returns (pcos_diagnosis, thyroid_dysfunction)."""
    pcos_diagnosis = joint_class // 2
    thyroid_dysfunction = joint_class % 2
    return pcos_diagnosis, thyroid_dysfunction


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
    """Wraps any results dict with a metadata block before saving, so
    anyone opening the file later can tell WHEN and on WHAT SETUP it was
    generated -- catches staleness and cross-machine drift at a glance."""
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