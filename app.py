import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from openproctor import Pipeline

st.set_page_config(
    page_title="OpenProctor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .infraction-card {
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .keyword-badge {
        display: inline-block;
        background: #ff4b4b;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stStatus { margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Reset state on new session
# ---------------------------------------------------------------------------
for key in ("report", "video_name", "pipeline_done"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ OpenProctor")
    st.caption("Detección de trampas en grabaciones de exámenes")

    uploaded = st.file_uploader(
        "Subir grabación (.mp4)",
        type=["mp4"],
        help="Archivo MP4 de la sesión de examen (Veyon u otro)",
    )

    with st.expander("⚙️ Configuración", expanded=False):
        jump_sec = st.slider("Salto entre frames (seg)", 1, 30, 5, help="A mayor salto, más rápido pero menos preciso")
        vlm_model = st.selectbox(
            "Modelo VLM (Ollama)",
            ["moondream", "llava", "minicpm-v", "bakllava"],
            index=0,
        )
        ocr_gpu = st.checkbox("GPU para OCR", value=True)

    run_btn = st.button(
        "▶ Ejecutar Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

    if uploaded is not None:
        st.info(f"Archivo: **{uploaded.name}**\n\n{uploaded.size / 1024 / 1024:.1f} MB")

    st.divider()
    st.caption("OpenProctor v0.1 — MIT License")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🔍 Panel de Auditoría")

if not uploaded:
    st.info("Sube un archivo .mp4 en el panel izquierdo y presiona **Ejecutar Pipeline**.")
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_btn:
    st.session_state.report = None
    st.session_state.pipeline_done = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name

    for d in ["data/interim", "data/suspects"]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)

    pipeline = Pipeline(
        video_path=video_path,
        jump_sec=jump_sec,
        ocr_gpu=ocr_gpu,
        vlm_model=vlm_model,
    )

    status_placeholder = st.empty()

    try:
        with st.status("🚀 Iniciando pipeline …", expanded=True) as status:

            def on_progress(phase: str, pct: float, msg: str):
                icons = {"extraction": "🎬", "ocr": "🔍", "vlm": "🧠"}
                icon = icons.get(phase, "⚙️")
                status.update(label=f"{icon} **{msg}**")
                if pct >= 1.0:
                    status.update(state="complete")
                # show a per-phase progress bar below the status
                if phase == "extraction":
                    status_placeholder.progress(min(pct, 1.0), text="Extrayendo frames …")
                elif phase == "ocr":
                    status_placeholder.progress(min(pct, 1.0), text="Analizando con OCR …")
                elif phase == "vlm":
                    status_placeholder.progress(min(pct, 1.0), text="Consultando VLM …")

            report = pipeline.run(progress=on_progress)

        status_placeholder.empty()
        n = report.get("summary", {}).get("infractions_confirmed", 0)
        if n > 0:
            st.success(f"✅ Pipeline completado — {n} infracción(es) confirmada(s)")
        else:
            st.success("✅ Pipeline completado — sin infracciones detectadas")

        st.session_state.report = report
        st.session_state.video_name = uploaded.name
        st.session_state.pipeline_done = True

    except Exception as e:
        status_placeholder.empty()
        st.error(f"❌ Error en el pipeline: {e}")
        raise

    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.pipeline_done and st.session_state.report:
    report = st.session_state.report
    summary = report.get("summary", {})
    infractions = report.get("infractions", [])

    # --- Summary metrics ---
    cols = st.columns(4)
    cols[0].metric("🎬 Frames extraídos", summary.get("total_frames_extracted", 0))
    cols[1].metric("🔍 Sospechosos (OCR)", summary.get("suspects_found", 0))
    cols[2].metric("🧠 Analizados (VLM)", summary.get("suspects_found", 0))
    cols[3].metric(
        "🛑 Infracciones",
        summary.get("infractions_confirmed", 0),
    )

    st.divider()

    # --- Infraction cards ---
    if infractions:
        st.subheader(f"🛑 {len(infractions)} infracción(es) confirmada(s)")
        for i, inf in enumerate(infractions, 1):
            file_path = inf.get("file", "")
            ts = inf.get("timestamp", "")
            kw = inf.get("keyword", "")
            reason = inf.get("reason", "")
            ocr_text = inf.get("ocr_text", "")

            with st.container(border=True):
                img_col, meta_col = st.columns([1, 1.8])
                with img_col:
                    if os.path.exists(file_path):
                        st.image(file_path, use_container_width=True)
                    else:
                        st.caption("(imagen no disponible)")
                with meta_col:
                    st.markdown(f"### #{i} — ⏱ `{ts}`")
                    if kw:
                        st.markdown(f"<span class='keyword-badge'>{kw}</span>",
                                    unsafe_allow_html=True)
                    st.markdown(f"**🧠 Motivo de la IA:**  \n{reason}")
                    if ocr_text:
                        with st.expander("📝 Texto OCR detectado"):
                            st.text(ocr_text[:500])
    else:
        st.info("✅ No se detectaron infracciones en esta grabación.")

# ---------------------------------------------------------------------------
# Instructions panel when results are not ready
# ---------------------------------------------------------------------------
elif not st.session_state.pipeline_done and uploaded is not None:
    st.info("Presiona **Ejecutar Pipeline** en el panel izquierdo para comenzar el análisis.")
