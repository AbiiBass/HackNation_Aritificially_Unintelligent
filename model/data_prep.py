"""
data_prep.py
Step 1 of the Women's Hormonal Health ensemble model pipeline.

SOURCING DECISION (updated from the original NHANES-only plan):
NHANES turned out to be a dead end for this project -- CDC discontinued
the Thyroid Profile component after 2011-2012, so there's no cycle where
thyroid labs and a useful PCOS-relevant hormone panel exist on the same
people. Instead we use two REAL, LABELED, purpose-built datasets:

1. PCOS_data_without_infertility (Kaggle, Kottarathil / Prasoon, sourced
   from 10 hospitals in Kerala, India; 541 women, 44 features).
   - Has a real clinician-assigned "PCOS (Y/N)" label.
   - ALSO has "TSH (mIU/L)" measured on the same patients -- so we get
     real joint (PCOS x thyroid) signal from ONE cohort, instead of
     stitching together two populations that never overlap.
   - Official source: https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos
   - This script falls back to a GitHub mirror of the same file if you
     haven't downloaded it yourself (see DOWNLOAD NOTES below).

2. UCI "Hypothyroid" dataset (aka Garvan Institute thyroid data), ~3,160
   patients, MIXED SEX, with a real hypothyroid/negative diagnosis label.
   - This is our stand-in for "male-skewed": it's a generic, sex-blind
     thyroid model trained on a population that is NOT female-specific,
     representing the historical baseline these tools were built on.
   - Official source: https://archive.ics.uci.edu/dataset/102/thyroid+disease
     (the "hypothyroid.csv" variant specifically)

DOWNLOAD NOTES:
- Best practice for your final submission: download both datasets
  yourself from the official links above and cite them properly in your
  docs. Kaggle requires a free account + API token (kaggle.json) to
  download via code, or you can just click "Download" in the browser.
- For convenience during development, this script will automatically
  fall back to fetching a GitHub-hosted mirror of each file if it can't
  find a local copy in data/raw/. Mirrors are useful for prototyping but
  aren't a citable source -- swap in the official download before your
  final submission.
------------------------------------------------------------------------
"""

import os
import io
import urllib.request
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

PCOS_LOCAL_CANDIDATES = [
    "PCOS_data_without_infertility.xlsx",
    "PCOS_data_without_infertility.csv",
]
PCOS_MIRROR_URL = (
    "https://raw.githubusercontent.com/kapuriajeet/"
    "PCOS-Detection-using-Machine-Learning/main/"
    "PCOS_data_without_infertility.xlsx"
)

THYROID_LOCAL_CANDIDATES = ["hypothyroid.data", "hypothyroid.csv"]
THYROID_MIRROR_URL = (
    "https://raw.githubusercontent.com/luca-scr/"
    "SGMM_Class_Imbalance/master/hypothyroid.csv"
)

# The official UCI .data file has NO header row -- columns are documented
# but not present in the file itself. This is the fixed column order.
THYROID_UCI_COLUMNS = [
    "diagnosis", "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "thyroid_surgery", "query_hypothyroid",
    "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
]


def _find_local(candidates):
    for name in candidates:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _fetch_bytes(url: str) -> bytes:
    print("  " + "!" * 66)
    print("  ! REPRODUCIBILITY WARNING: fetching from an unofficial GitHub  !")
    print("  ! mirror, not the official Kaggle/UCI source. Fine for dev,   !")
    print("  ! but NOT citable for your final submission -- download the  !")
    print("  ! official files yourself and place them in data/raw/ before !")
    print("  ! submitting (see this file's docstring for official links). !")
    print("  " + "!" * 66)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


# ----------------------------------------------------------------------
# 1. PCOS cohort (real PCOS label + TSH, same patients)
# ----------------------------------------------------------------------
def load_pcos_raw() -> pd.DataFrame:
    local = _find_local(PCOS_LOCAL_CANDIDATES)
    if local:
        print(f"[pcos] using local file: {local}")
        if local.endswith(".xlsx"):
            return pd.read_excel(local, sheet_name="Full_new")
        return pd.read_csv(local)

    print(f"[pcos] no local file found in {RAW_DIR}/, fetching mirror...")
    raw_bytes = _fetch_bytes(PCOS_MIRROR_URL)
    return pd.read_excel(io.BytesIO(raw_bytes), sheet_name="Full_new")


def clean_pcos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace from column names -- the source file is inconsistent
    # about leading/trailing spaces (e.g. " Age (yrs)", "Height(Cm) ")
    df.columns = [c.strip() for c in df.columns]

    # Drop the stray unnamed trailing column and ID columns we don't need
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop(columns=["Sl. No", "Patient File No."], errors="ignore")

    rename_map = {
        "PCOS (Y/N)": "pcos_diagnosis",
        "Age (yrs)": "age",
        "Weight (Kg)": "weight_kg",
        "Height(Cm)": "height_cm",
        "BMI": "bmi",
        "Cycle(R/I)": "cycle_regularity",       # 2 = regular, 4 = irregular (per source coding)
        "Cycle length(days)": "cycle_length_days",
        "FSH(mIU/mL)": "fsh",
        "LH(mIU/mL)": "lh",
        "FSH/LH": "fsh_lh_ratio",
        "TSH (mIU/L)": "tsh",
        "AMH(ng/mL)": "amh",
        "PRL(ng/mL)": "prl",
        "Vit D3 (ng/mL)": "vit_d3",
        "PRG(ng/mL)": "progesterone",
        "RBS(mg/dl)": "random_blood_sugar",
        "Weight gain(Y/N)": "weight_gain",
        "hair growth(Y/N)": "hirsutism",
        "Skin darkening (Y/N)": "skin_darkening",
        "Hair loss(Y/N)": "hair_loss",
        "Pimples(Y/N)": "acne",
        "Fast food (Y/N)": "fast_food",
        "Reg.Exercise(Y/N)": "regular_exercise",
        "BP _Systolic (mmHg)": "bp_systolic",
        "BP _Diastolic (mmHg)": "bp_diastolic",
        "Follicle No. (L)": "follicle_count_left",
        "Follicle No. (R)": "follicle_count_right",
        "Avg. F size (L) (mm)": "avg_follicle_size_left",
        "Avg. F size (R) (mm)": "avg_follicle_size_right",
        "Endometrium (mm)": "endometrium_mm",
    }
    df = df.rename(columns=rename_map)

    # AMH sometimes arrives as strings (source data quirk) -- coerce to numeric
    for col in ["amh", "tsh", "fsh", "lh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sex"] = "female"  # entire cohort is female by construction
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Mixed-sex thyroid cohort (male-skewed baseline)
# ----------------------------------------------------------------------
def load_thyroid_raw() -> pd.DataFrame:
    local = _find_local(THYROID_LOCAL_CANDIDATES)
    if local:
        print(f"[thyroid] using local file: {local}")
        raw_bytes = open(local, "rb").read()
    else:
        print(f"[thyroid] no local file found in {RAW_DIR}/, fetching mirror...")
        raw_bytes = _fetch_bytes(THYROID_MIRROR_URL)

    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = [l for l in text.splitlines() if l.strip()]

    # Two possible formats we might see:
    #  (a) official UCI .data: no header, no comments, just data rows
    #  (b) some mirrors: '#'-comment block, then a header line, then data
    non_comment_lines = [l for l in lines if not l.strip().startswith("#")]
    first_field = non_comment_lines[0].split(",")[0].strip().lower()

    if first_field == "diagnosis":
        # format (b): first non-comment line IS the header
        csv_text = "\n".join(non_comment_lines)
        df = pd.read_csv(io.StringIO(csv_text), sep=r"\s*,\s*", engine="python")
    else:
        # format (a): no header anywhere -- assign the known UCI column order
        csv_text = "\n".join(non_comment_lines)
        df = pd.read_csv(
            io.StringIO(csv_text), sep=r"\s*,\s*", engine="python",
            header=None, names=THYROID_UCI_COLUMNS,
        )
    return df


def clean_thyroid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    keep = ["diagnosis", "age", "sex", "pregnant", "sick", "TSH", "T3", "TT4", "T4U", "FTI"]
    df = df[[c for c in keep if c in df.columns]]

    df = df.rename(columns={
        "diagnosis": "thyroid_diagnosis",  # 'hypothyroid' or 'negative'
        "TSH": "tsh",
        "T3": "t3",
        "TT4": "t4",
        "T4U": "t4_uptake",
        "FTI": "free_thyroxine_index",
    })

    # Source uses '?' for missing values and M/F for sex
    df = df.replace("?", pd.NA)
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"M": "male", "F": "female"})
    for col in ["age", "tsh", "t3", "t4", "t4_uptake", "free_thyroxine_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["age"].between(0, 120, inclusive="both") | df["age"].isna()]
    df["thyroid_diagnosis_binary"] = (df["thyroid_diagnosis"] == "hypothyroid").astype(int)

    return df.reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pcos_raw = load_pcos_raw()
    pcos_clean = clean_pcos(pcos_raw)
    pcos_out = os.path.join(OUT_DIR, "pcos_cohort.csv")
    pcos_clean.to_csv(pcos_out, index=False)
    print(f"\n[pcos] saved {len(pcos_clean)} rows x {len(pcos_clean.columns)} cols -> {pcos_out}")
    print(f"[pcos] PCOS positive: {pcos_clean['pcos_diagnosis'].sum()} / {len(pcos_clean)}")

    thyroid_raw = load_thyroid_raw()
    thyroid_clean = clean_thyroid(thyroid_raw)
    thyroid_out = os.path.join(OUT_DIR, "thyroid_mixed_cohort.csv")
    thyroid_clean.to_csv(thyroid_out, index=False)
    print(f"\n[thyroid] saved {len(thyroid_clean)} rows x {len(thyroid_clean.columns)} cols -> {thyroid_out}")
    print(f"[thyroid] hypothyroid positive: {thyroid_clean['thyroid_diagnosis_binary'].sum()} / {len(thyroid_clean)}")
    print(f"[thyroid] sex distribution:\n{thyroid_clean['sex'].value_counts()}")


if __name__ == "__main__":
    main()