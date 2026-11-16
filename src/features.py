"""Persiapan fitur diagnosis bearing: sama dengan tahap training notebook Kasus 04."""

import numpy as np
import pandas as pd

TARGET = "kondisi"
ID_COLUMN = "id_rekaman"
CATEGORY_COLUMN = "posisi_sensor"
NUMERIC_COLUMNS = [
    "kecepatan_rpm", "beban_kN", "suhu_housing_C", "rms_g", "puncak_g",
    "kurtosis", "crest_factor", "energi_bpfo", "energi_bpfi",
    "energi_bsf", "frekuensi_dominan_hz",
]
RAW_FEATURES = [CATEGORY_COLUMN, *NUMERIC_COLUMNS]
ENGINEERED_COLUMNS = [
    "energi_total", "rasio_bpfo", "rasio_bpfi", "rasio_bsf",
    "frekuensi_per_putaran", "puncak_dibagi_rms",
]
LABELS = ["normal", "outer_race", "inner_race", "bola"]
DISPLAY_LABELS = {
    "normal": "Normal", "outer_race": "Kerusakan cincin luar",
    "inner_race": "Kerusakan cincin dalam", "bola": "Kerusakan bola",
}
SPARE_PARTS = {
    "normal": "Tidak ada komponen yang perlu diganti berdasarkan prediksi ini; lanjutkan inspeksi berkala.",
    "outer_race": "Siapkan bearing pengganti dengan spesifikasi cincin luar yang sesuai dan perlengkapan pemasangan.",
    "inner_race": "Siapkan bearing pengganti sesuai ukuran cincin dalam/poros dan perlengkapan pemasangan.",
    "bola": "Siapkan bearing pengganti lengkap yang kompatibel dan perlengkapan pemasangan.",
}
ALIASES = {
    "normal": "normal", "ok": "normal",
    "outer_race": "outer_race", "outer": "outer_race", "or": "outer_race",
    "inner_race": "inner_race", "inner": "inner_race", "ir": "inner_race",
    "bola": "bola", "ball": "bola",
}

def normalize_labels(y: pd.Series) -> pd.Series:
    raw = y.astype("string").str.strip().str.lower().str.replace(r"[\s-]+", "_", regex=True)
    mapped = raw.map(ALIASES)
    bad = sorted(raw[mapped.isna()].dropna().unique().tolist())
    if bad:
        raise ValueError(f"Label target belum dikenal: {bad}")
    return mapped.astype(str)

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(RAW_FEATURES + [ID_COLUMN, TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Kolom data hilang: {missing}")
    before = len(df)
    out = df.drop_duplicates().copy()
    duplicate_count = before - len(out)
    out[TARGET] = normalize_labels(out[TARGET])
    out[CATEGORY_COLUMN] = out[CATEGORY_COLUMN].astype("string").str.strip().str.lower()
    problems = {
        "rows_raw": before,
        "exact_duplicates": duplicate_count,
        "rows_clean": len(out),
        "missing_suhu": int(out["suhu_housing_C"].isna().sum()),
        "missing_kurtosis": int(out["kurtosis"].isna().sum()),
        "rpm_zero_or_negative": int((pd.to_numeric(out["kecepatan_rpm"], errors="coerce") <= 0).sum()),
        "rms_zero_or_negative": int((pd.to_numeric(out["rms_g"], errors="coerce") <= 0).sum()),
        "unique_ids": int(out[ID_COLUMN].nunique()),
        "class_counts": out[TARGET].value_counts().to_dict(),
    }
    return out, problems

def prepare_features(df: pd.DataFrame, engineered: bool = False) -> pd.DataFrame:
    missing = sorted(set(RAW_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Kolom fitur hilang: {missing}")
    x = df[RAW_FEATURES].copy()
    x[CATEGORY_COLUMN] = x[CATEGORY_COLUMN].astype("string").str.strip().str.lower().fillna("unknown")
    x[CATEGORY_COLUMN] = x[CATEGORY_COLUMN].where(x[CATEGORY_COLUMN].isin(["drive_end", "fan_end"]), "unknown")
    for column in NUMERIC_COLUMNS:
        x[column] = pd.to_numeric(x[column], errors="coerce")
        x[column] = x[column].replace([np.inf, -np.inf], np.nan)

    x.loc[x["kecepatan_rpm"] <= 0, "kecepatan_rpm"] = np.nan
    x.loc[x["rms_g"] <= 0, "rms_g"] = np.nan
    for col in ["beban_kN", "puncak_g", "kurtosis", "crest_factor", "energi_bpfo", "energi_bpfi", "energi_bsf", "frekuensi_dominan_hz"]:
        x.loc[x[col] < 0, col] = np.nan
    if engineered:
        e = x["energi_bpfo"] + x["energi_bpfi"] + x["energi_bsf"]
        x["energi_total"] = e
        safe_e = e.replace(0, np.nan)
        x["rasio_bpfo"] = x["energi_bpfo"] / safe_e
        x["rasio_bpfi"] = x["energi_bpfi"] / safe_e
        x["rasio_bsf"] = x["energi_bsf"] / safe_e
        x["frekuensi_per_putaran"] = 60 * x["frekuensi_dominan_hz"] / x["kecepatan_rpm"]
        x["puncak_dibagi_rms"] = x["puncak_g"] / x["rms_g"]
        x = x.replace([np.inf, -np.inf], np.nan)
    return x
