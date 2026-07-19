"""
train.py
Single entry point that reproduces the ENTIRE pipeline, in order.

POST-REVIEW FIX: this file used to be empty (0 bytes) while the real
training logic was scattered across 8 separate scripts with no
documented run order anywhere in the repo itself -- reproducing the
project required already knowing, from outside the codebase, which
scripts to run and in what sequence. Running this file now IS that
documentation, executably.

Usage:
    python3 train.py              # runs everything
    python3 train.py --skip-data  # skips data_prep.py (e.g. if you
                                    already have data/*.csv and don't
                                    want to re-fetch/re-download)

Each step is a separate subprocess (not an import) so a failure in one
step prints a clear traceback and stops the whole run, rather than
silently continuing with stale in-memory state.
"""

import subprocess
import sys
import argparse

STEPS = [
    ("data_prep.py", "Step 1: load PCOS + thyroid cohorts"),
    ("label_engineering.py", "Step 2: build labels (thyroid proxy, joint 4-class target)"),
    ("split.py", "Step 3: stratified train/val/test split"),
    ("train_model1.py", "Step 4a: train Model 1 (generic clinical baseline)"),
    ("train_model2.py", "Step 4b: train Model 2 (women-only, thyroid axis)"),
    ("train_model2b_pcos.py", "Step 4c: train Model 2b (women-only, PCOS axis)"),
    ("model3.py", "Step 5: Model 3 (clinical criteria, no training)"),
    ("meta_learner.py", "Step 6: train the meta-learner"),
    ("robustness_checks.py", "Step 7: ablation, bootstrap CIs, k-fold CV, heuristic stress test"),
    ("label_noise_sensitivity.py", "Step 8: label-noise sensitivity analysis"),
]


def run_step(script: str, description: str):
    print("\n" + "=" * 70)
    print(f"{description}")
    print(f"  ({script})")
    print("=" * 70)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n!! FAILED at {script} (exit code {result.returncode}) -- stopping pipeline.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full Women's Hormonal Health model pipeline.")
    parser.add_argument("--skip-data", action="store_true", help="Skip data_prep.py")
    args = parser.parse_args()

    steps = STEPS
    if args.skip_data:
        steps = [s for s in steps if s[0] != "data_prep.py"]
        print("(--skip-data: assuming data/*.csv already exist from a previous run)")

    for script, description in steps:
        run_step(script, description)

    print("\n" + "=" * 70)
    print("Pipeline complete. All models saved to models/, all reports saved")
    print("as timestamped JSON (see each file's _metadata block) to models/.")
    print("=" * 70)


if __name__ == "__main__":
    main()