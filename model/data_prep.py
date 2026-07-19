"""
data_prep.py
Loads and cleans two datasets: PCOS_data_without_infertility (Kaggle, 541
women, real PCOS label + TSH on the same patients) and UCI Hypothyroid
(~3160 patients, mixed sex, real diagnosis label, generic baseline).

Falls back to a GitHub mirror if local files aren't in data/raw/ -- not
citable, use the official source for final submission.
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

# UCI .data file has no header row; this is the fixed column order.
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
# PCOS cohort (real PCOS label + TSH, same patients)
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
    # Strip whitespace from column names (source file has inconsistent spacing)
    df.columns = [c.strip() for c in df.columns]

    # Drop unused unnamed/ID columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop(columns=["Sl. No", "Patient File No."], errors="ignore")

    rename_map = {
        "PCOS (Y/N)": "pcos_diagnosis",
        "Age (yrs)": "age",
        "Weight (Kg)": "weight_kg",
        "Height(Cm)": "height_cm",
        "BMI": "bmi",
        "Cycle(R/I)": "cycle_regularity",       # 2 = regular, 4 = irregular
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

    # AMH sometimes arrives as strings -- coerce to numeric
    for col in ["amh", "tsh", "fsh", "lh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sex"] = "female"  # entire cohort is female
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Mixed-sex thyroid cohort
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

    # Two possible formats: (a) UCI .data -- no header; (b) mirrors -- comment
    # block then a header line
    non_comment_lines = [l for l in lines if not l.strip().startswith("#")]
    first_field = non_comment_lines[0].split(",")[0].strip().lower()

    if first_field == "diagnosis":
        # format (b): header present
        csv_text = "\n".join(non_comment_lines)
        df = pd.read_csv(io.StringIO(csv_text), sep=r"\s*,\s*", engine="python")
    else:
        # format (a): no header, use known UCI column order
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