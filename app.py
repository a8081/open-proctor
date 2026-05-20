import json
import os
import shutil
from pathlib import Path

import streamlit as st

from openproctor import Pipeline

st.set_page_config(
    page_title="OpenProctor - Lote",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .video-row {
        display: flex; align-items: center; gap: 1rem;
        padding: 0.6rem 0; border-bottom: 1px solid #eee;
    }
    .badge-pending {
        background: #f0f0f0; color: #666;
        padding: 2px 12px; border-radius: 12px; font-size: 0.8rem;
    }
    .badge-ok {
        background: #d4edda; color: #155724;
        padding: 2px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    .keyword-badge {
        display: inline-block; background: #ff4b4b; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

INPUT_DIR = Path("data/input")
REPORTS_DIR = Path("data/reports")
INTERIM_BASE = Path("data/interim")
SUSPECTS_BASE = Path("data/suspects")

for d in (INPUT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key in ("selected_video", "batch_active", "batch_results"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scan_videos():
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    rows = []
    for v in videos:
        report_path = REPORTS_DIR / f"{v.stem}_report.json"
        completed = report_path.exists()
        rows.append({
            "path": v,
            "name": v.name,
            "size_mb": v.stat().st_size / (1024 * 1024),
            "completed": completed,
            "report_path": report_path if completed else None,
        })
    return rows


def _load_report(report_path: Path) -> dict:
    if report_path and report_path.exists():
        return json.loads(report_path.read_text())
    return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ OpenProctor")
    st.caption("Auditoría de exámenes por lotes")

    uploaded = st.file_uploader(
        "Subir .mp4 (se guarda en data/input/)",
        type=["mp4"],
    )
    if uploaded is not None:
        dst = INPUT_DIR / uploaded.name
        dst.write_bytes(uploaded.getvalue())
        st.success(f"{uploaded.name} guardado en data/input/")
        st.rerun()

    st.divider()

    with st.expander("⚙️ Configuración", expanded=False):
        jump_sec = st.slider("Salto entre frames (seg)", 1, 30, 5)
        vlm_model = st.selectbox("Modelo VLM", ["moondream", "llava", "minicpm-v", "bakllava"])
        ocr_gpu = st.checkbox("GPU para OCR", value=True)

    st.divider()
    st.caption("OpenProctor v0.1")

# ---------------------------------------------------------------------------
# Main area — Video list
# ---------------------------------------------------------------------------
st.title("📂 Videos en data/input/")

videos = _scan_videos()

if not videos:
    st.info("No hay archivos .mp4 en `data/input/`.  \n  "
            "Cópialos manualmente o usa el panel izquierdo para subir uno.")
    st.stop()

# --- Table header ---
cols = st.columns([3, 1, 1.5, 1.5])
cols[0].markdown("**Video**")
cols[1].markdown("**Tamaño**")
cols[2].markdown("**Estado**")
cols[3].markdown("**Acción**")
st.markdown("---")

# --- Table rows ---
selected_name = None
for v in videos:
    c1, c2, c3, c4 = st.columns([3, 1, 1.5, 1.5])
    c1.markdown(f"`{v['name']}`")
    c2.markdown(f"{v['size_mb']:.0f} MB")

    if v["completed"]:
        c3.markdown("<span class='badge-ok'>✅ Completado</span>", unsafe_allow_html=True)
        if c4.button("📄 Ver", key=f"view_{v['name']}"):
            st.session_state.selected_video = v["name"]
            st.rerun()
    else:
        c3.markdown("<span class='badge-pending'>⏳ Pendiente</span>", unsafe_allow_html=True)
        c4.markdown("—")

st.markdown("---")

# --- Batch control ---
pending = [v for v in videos if not v["completed"]]
total = len(videos)
done = total - len(pending)

c_progress, c_btn = st.columns([3, 1])
c_progress.markdown(f"**Procesados:** {done} / {total}")
batch_btn = c_btn.button(
    f"▶ Procesar pendientes ({len(pending)})",
    type="primary",
    use_container_width=True,
    disabled=len(pending) == 0,
)

# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------
if batch_btn:
    st.session_state.batch_active = True
    status_placeholder = st.empty()
    phase_placeholder = st.empty()
    progress_bar = st.progress(0)
    results_placeholder = st.empty()

    for idx, v in enumerate(pending):
        video = v["path"]
        name = v["name"]
        report_path = REPORTS_DIR / f"{video.stem}_report.json"

        # Clean per-video temp dirs from previous runs
        for d in (INTERIM_BASE / video.stem, SUSPECTS_BASE / video.stem):
            if d.exists():
                shutil.rmtree(d)

        pipeline = Pipeline(
            video_path=video,
            jump_sec=jump_sec,
            ocr_gpu=ocr_gpu,
            vlm_model=vlm_model,
        )

        overall_progress = (done + idx) / total

        with st.status(
            f"({done + idx + 1}/{total}) {name}",
            expanded=True,
        ) as status:

            def make_progress_cb(status, phase_placeholder):
                def cb(phase, pct, msg):
                    icons = {"extraction": "🎬", "ocr": "🔍", "vlm": "🧠"}
                    status.update(label=f"({done + idx + 1}/{total}) {name}  —  {icons.get(phase, '')} {msg}")
                    phase_placeholder.progress(
                        overall_progress + (pct / total),
                        text=f"{icons.get(phase, '')} [{phase}] {msg}",
                    )
                return cb

            progress_bar.progress(overall_progress, text=f"Procesando {name} …")
            cb = make_progress_cb(status, phase_placeholder)

            try:
                report = pipeline.run(progress=cb)
                status.update(
                    label=f"✅ ({done + idx + 1}/{total}) {name} — {report['summary']['infractions_confirmed']} infracción(es)",
                    state="complete",
                )
            except Exception as e:
                status.update(
                    label=f"❌ ({done + idx + 1}/{total}) {name} — Error: {e}",
                    state="error",
                )

        progress_bar.progress((done + idx + 1) / total)

    progress_bar.empty()
    phase_placeholder.empty()
    st.session_state.batch_active = False
    st.success(f"✅ Lote completado — {total} video(s) procesado(s)")
    st.rerun()

# ---------------------------------------------------------------------------
# Detail view — selected completed video
# ---------------------------------------------------------------------------
if st.session_state.selected_video:
    sel = st.session_state.selected_video
    report_path = REPORTS_DIR / f"{Path(sel).stem}_report.json"

    if report_path.exists():
        report = _load_report(report_path)
        summary = report.get("summary", {})
        infractions = report.get("infractions", [])

        st.divider()
        col_back, col_title = st.columns([0.1, 0.9])
        with col_back:
            if st.button("← Volver"):
                st.session_state.selected_video = None
                st.rerun()
        with col_title:
            st.subheader(f"📄 {sel}")

        # Metrics
        mc = st.columns(4)
        mc[0].metric("🎬 Frames extraídos", summary.get("total_frames_extracted", 0))
        mc[1].metric("🔍 Sospechosos (OCR)", summary.get("suspects_found", 0))
        mc[2].metric("🧠 Analizados (VLM)", summary.get("suspects_found", 0))
        mc[3].metric("🛑 Infracciones", summary.get("infractions_confirmed", 0))

        if infractions:
            st.divider()
            st.subheader(f"🛑 {len(infractions)} infracción(es) confirmada(s)")
            for i, inf in enumerate(infractions, 1):
                file_path = inf.get("file", "")
                ts = inf.get("timestamp", "")
                kw = inf.get("keyword", "")
                reason = inf.get("reason", "")
                ocr_text = inf.get("ocr_text", "")

                with st.container(border=True):
                    ic, mc2 = st.columns([1, 1.8])
                    with ic:
                        if os.path.exists(file_path):
                            st.image(file_path, use_container_width=True)
                        else:
                            st.caption("(imagen no disponible)")
                    with mc2:
                        st.markdown(f"### #{i} — ⏱ `{ts}`")
                        if kw:
                            st.markdown(
                                f"<span class='keyword-badge'>{kw}</span>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"**🧠 Motivo de la IA:**  \n{reason}")
                        if ocr_text:
                            with st.expander("📝 Texto OCR detectado"):
                                st.text(ocr_text[:500])
        else:
            st.info("✅ No se detectaron infracciones en esta grabación.")

    else:
        st.warning(f"Reporte no encontrado: {report_path}")
        if st.button("← Volver"):
            st.session_state.selected_video = None
            st.rerun()
