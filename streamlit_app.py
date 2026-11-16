from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from src.features import (
    DISPLAY_LABELS,
    ENGINEERED_COLUMNS,
    LABELS,
    RAW_FEATURES,
    SPARE_PARTS,
    prepare_features,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"

st.set_page_config(page_title="BearingCare", page_icon="⚙️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def predict_records(data: pd.DataFrame, model, meta: dict) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        engineered = any(col in list(expected) for col in ENGINEERED_COLUMNS)
    else:
        engineered = bool(meta.get("engineered_features", True))

    features = prepare_features(data, engineered=engineered)
    pred = model.predict(features)
    proba = model.predict_proba(features)
    result = data.copy().reset_index(drop=True)
    result["prediksi_kondisi"] = pred
    for label in LABELS:
        if label in model.classes_:
            position = list(model.classes_).index(label)
            result[f"prob_{label}"] = proba[:, position]
        else:
            result[f"prob_{label}"] = 0.0
    return result


st.title("⚙️ BearingCare")
st.caption("Diagnosis awal kondisi bearing berdasarkan ringkasan rekaman getaran 1 detik")

if not MODEL_PATH.exists():
    st.error("File model belum ditemukan: artifacts/model.joblib")
    st.write("Unduh hasil training dari Google Colab, lalu simpan sebagai `artifacts/model.joblib` pada repository proyek.")
    st.stop()

try:
    model = load_model()
    metadata = load_metadata()
except Exception as error:
    st.error(f"Model tidak dapat dibuka: {error}")
    st.caption("Periksa versi scikit-learn di requirements.txt dan file model hasil training.")
    st.stop()

with st.sidebar:
    st.subheader("Informasi model")
    st.write("Model:", metadata.get("selected_model", type(model.named_steps.get("model", model)).__name__ if hasattr(model, "named_steps") else type(model).__name__))
    st.write("Versi:", metadata.get("version", "–"))
    test_scores = metadata.get("test_results", {})
    selected = metadata.get("selected_model", "")
    score = test_scores.get(selected, {}).get("macro_f1")
    if score is not None:
        st.metric("Macro F1 (test)", f"{score:.4f}")
    st.caption("Hasil prediksi perlu dikonfirmasi dengan inspeksi teknisi.")

single_tab, batch_tab = st.tabs(["Satu rekaman", "Prediksi CSV"])

with single_tab:
    with st.form("form_bearing"):
        col1, col2, col3 = st.columns(3)
        with col1:
            posisi_sensor = st.selectbox("Posisi sensor", ["drive_end", "fan_end"])
            kecepatan_rpm = st.number_input("Kecepatan (RPM)", min_value=1.0, value=2095.5, step=1.0)
            beban_kN = st.number_input("Beban (kN)", min_value=0.0, value=3.23, format="%.2f")
            suhu_housing_C = st.number_input("Suhu housing (°C)", value=52.7, format="%.1f")
        with col2:
            rms_g = st.number_input("RMS getaran (g)", min_value=0.0001, value=0.4695, format="%.4f")
            puncak_g = st.number_input("Getaran puncak (g)", min_value=0.0001, value=1.5433, format="%.4f")
            kurtosis = st.number_input("Kurtosis", min_value=0.0, value=3.3285, format="%.4f")
            crest_factor = st.number_input("Crest factor", min_value=0.0, value=3.314, format="%.3f")
        with col3:
            energi_bpfo = st.number_input("Energi BPFO", min_value=0.0, value=0.0804, format="%.4f")
            energi_bpfi = st.number_input("Energi BPFI", min_value=0.0, value=0.0659, format="%.4f")
            energi_bsf = st.number_input("Energi BSF", min_value=0.0, value=0.0614, format="%.4f")
            frekuensi_dominan_hz = st.number_input("Frekuensi dominan (Hz)", min_value=0.0, value=49.0, format="%.1f")
        submitted = st.form_submit_button("Analisis bearing", type="primary")

    if submitted:
        if puncak_g < rms_g:
            st.error("Getaran puncak tidak boleh lebih kecil daripada RMS.")
        else:
            values = {
                "posisi_sensor": posisi_sensor,
                "kecepatan_rpm": kecepatan_rpm,
                "beban_kN": beban_kN,
                "suhu_housing_C": suhu_housing_C,
                "rms_g": rms_g,
                "puncak_g": puncak_g,
                "kurtosis": kurtosis,
                "crest_factor": crest_factor,
                "energi_bpfo": energi_bpfo,
                "energi_bpfi": energi_bpfi,
                "energi_bsf": energi_bsf,
                "frekuensi_dominan_hz": frekuensi_dominan_hz,
            }
            try:
                result = predict_records(pd.DataFrame([values]), model, metadata).iloc[0]
                label = result["prediksi_kondisi"]
                title = DISPLAY_LABELS[label]
                if label == "normal":
                    st.success(f"Hasil prediksi: {title}")
                else:
                    st.warning(f"Hasil prediksi: {title}")
                st.write("**Persiapan suku cadang:**", SPARE_PARTS[label])
                chart = pd.DataFrame({
                    "Kondisi": [DISPLAY_LABELS[k] for k in LABELS],
                    "Probabilitas": [float(result[f"prob_{k}"]) for k in LABELS],
                }).set_index("Kondisi")
                st.bar_chart(chart)
                with st.expander("Probabilitas numerik"):
                    st.dataframe(chart.style.format({"Probabilitas": "{:.2%}"}))
            except Exception as error:
                st.error(f"Prediksi gagal: {error}")

with batch_tab:
    st.write("Unggah file CSV yang mempunyai kolom input sama seperti dataset, tanpa kolom target `kondisi` pun tidak masalah.")
    upload = st.file_uploader("Pilih file CSV", type="csv")
    if upload is not None:
        try:
            records = pd.read_csv(upload)
            missing = sorted(set(RAW_FEATURES) - set(records.columns))
            if missing:
                st.error("Kolom belum lengkap: " + ", ".join(missing))
            elif records.empty:
                st.warning("File CSV kosong.")
            elif len(records) > 2000:
                st.error("Maksimal 2.000 rekaman dalam satu file.")
            else:
                output = predict_records(records, model, metadata)
                st.write(f"Berhasil memproses {len(output)} rekaman.")
                st.dataframe(output.head(50), use_container_width=True)
                st.download_button(
                    "Unduh hasil prediksi",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_bearing.csv",
                    mime="text/csv",
                )
        except Exception as error:
            st.error(f"File tidak dapat diproses: {error}")

st.caption("Aplikasi ini merupakan prototipe pembelajaran; hasil model bukan pengganti pemeriksaan mekanis.")
