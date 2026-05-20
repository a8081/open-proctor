import json
import os
import re
from pathlib import Path

import streamlit as st

from openproctor import Pipeline
from openproctor.i18n import I18n
from openproctor.vlm.ollama_client import VLMConfig, DEFAULT_MODELS

st.set_page_config(
    page_title="OpenProctor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .video-row { display: flex; align-items: center; gap: 1rem; padding: 0.6rem 0; border-bottom: 1px solid #eee; }
    .badge-pending { background: #f0f0f0; color: #666; padding: 2px 12px; border-radius: 12px; font-size: 0.8rem; }
    .badge-ok { background: #d4edda; color: #155724; padding: 2px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .keyword-badge { display: inline-block; background: #ff4b4b; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

INPUT_DIR = Path("data/input")
REPORTS_DIR = Path("data/reports")
INTERIM_BASE = Path("data/interim")
SUSPECTS_BASE = Path("data/suspects")
BATCH_STATE_FILE = Path("data/.batch_state.json")

for d in (INPUT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
class StopRequested(Exception):
    pass


_DEFAULTS = {
    "selected_video": None,
    "locale": "en",
    "batch_running": False,
    "batch_queue": [],
    "batch_queue_paths": [],
    "batch_done": 0,
    "batch_total": 0,
    "batch_processed": [],
    "batch_stop": False,
    "batch_stopped_msg": False,
    "batch_stopping_msg": False,
    "batch_completed_msg": False,
    "batch_retries": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# I18n
# ---------------------------------------------------------------------------
i18n = I18n(st.session_state.locale)

# ---------------------------------------------------------------------------
# Batch state persistence (survives browser refreshes)
# ---------------------------------------------------------------------------
def _save_batch_state():
    data = {
        "batch_running": st.session_state.batch_running,
        "batch_queue_paths": [str(v["path"]) for v in st.session_state.batch_queue],
        "batch_done": st.session_state.batch_done,
        "batch_total": st.session_state.batch_total,
        "batch_processed": list(st.session_state.batch_processed),
        "locale": st.session_state.locale,
        "batch_retries": dict(st.session_state.batch_retries),
    }
    BATCH_STATE_FILE.write_text(json.dumps(data, indent=2))


def _clear_batch_state():
    BATCH_STATE_FILE.unlink(missing_ok=True)


def _reconstruct_video(path: Path) -> dict:
    report_path = REPORTS_DIR / f"{path.stem}_report.json"
    completed = report_path.exists()
    return {
        "path": path,
        "name": path.name,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "completed": completed,
        "report_path": report_path if completed else None,
    }


# Restore persisted batch state on page load (e.g. after browser refresh)
if BATCH_STATE_FILE.exists() and not st.session_state.batch_running:
    try:
        saved = json.loads(BATCH_STATE_FILE.read_text())
        st.session_state.batch_running = saved.get("batch_running", False)
        st.session_state.batch_queue_paths = saved.get("batch_queue_paths", [])
        st.session_state.batch_queue = [
            _reconstruct_video(Path(p))
            for p in st.session_state.batch_queue_paths
            if Path(p).exists()
        ]
        st.session_state.batch_done = saved.get("batch_done", 0)
        st.session_state.batch_total = saved.get("batch_total", 0)
        st.session_state.batch_processed = saved.get("batch_processed", [])
        st.session_state.batch_retries = saved.get("batch_retries", {})
        locale = saved.get("locale", "en")
        st.session_state.locale = locale
        i18n.set_locale(locale)
    except Exception:
        _clear_batch_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scan_videos():
    rows = []
    for v in sorted(INPUT_DIR.glob("*.mp4")):
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


def _find_frames_dir(base: Path, stem: str) -> Path | None:
    candidates = sorted(base.glob(f"{stem}*"))
    for c in reversed(candidates):
        if c.is_dir() and any(True for _ in c.glob("*.jpg")):
            return c
    return None


def _parse_timestamp(name: str) -> str:
    m = re.search(r"min_(\d+)_seg_(\d+)", name)
    return f"{m.group(1)}m {m.group(2)}s" if m else name


def _show_frame_grid(frames: list[Path], key_prefix: str, cols_per_row: int = 3,
                      page_size: int = 15, findings_map: dict | None = None):
    if not frames:
        return
    _i18n = I18n(st.session_state.locale)

    total_pages = max(1, (len(frames) + page_size - 1) // page_size)
    page_key = f"{key_prefix}_page"
    page = st.session_state.get(page_key, 0)

    if page >= total_pages:
        page = total_pages - 1
        st.session_state[page_key] = page

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_info:
        st.markdown(
            f"<div style='text-align:center'><strong>{_i18n.t('page_info', page=page + 1, total=total_pages, count=len(frames))}</strong></div>",
            unsafe_allow_html=True,
        )
    with col_prev:
        if page > 0:
            if st.button(f"◀ {_i18n.t('previous')}", key=f"{key_prefix}_prev"):
                st.session_state[page_key] = page - 1
                st.rerun()
    with col_next:
        if page < total_pages - 1:
            if st.button(f"{_i18n.t('next')} ▶", key=f"{key_prefix}_next"):
                st.session_state[page_key] = page + 1
                st.rerun()

    start = page * page_size
    page_frames = frames[start:start + page_size]

    for row_start in range(0, len(page_frames), cols_per_row):
        row_frames = page_frames[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, fp in enumerate(row_frames):
            with cols[i]:
                ts = _parse_timestamp(fp.stem)
                st.caption(f"⏱ {ts}")
                suspect_info = None
                if findings_map:
                    entry = findings_map.get(fp.name)
                    if entry:
                        suspect_info = entry
                st.image(str(fp), use_container_width=True)
                if suspect_info:
                    if suspect_info.get("keywords"):
                        for kw in suspect_info["keywords"]:
                            st.markdown(
                                f"<span class='keyword-badge'>{kw}</span>",
                                unsafe_allow_html=True,
                            )
                    if suspect_info.get("ocr_text"):
                        with st.expander("📝 OCR"):
                            st.text(suspect_info["ocr_text"][:200])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(f"⚡ {i18n.t('app_title')}")
    st.caption(i18n.t("app_subtitle"))

    # Language selector — triggers immediate rerun
    lang = st.selectbox(
        i18n.t("language"),
        options=["en", "es"],
        format_func=lambda x: "English" if x == "en" else "Español",
        index=0 if st.session_state.locale == "en" else 1,
    )
    if lang != st.session_state.locale:
        st.session_state.locale = lang
        i18n.set_locale(lang)
        if BATCH_STATE_FILE.exists():
            _save_batch_state()  # persist language preference
        st.rerun()

    uploaded = st.file_uploader(i18n.t("upload_mp4"), type=["mp4"])
    if uploaded is not None:
        dst = INPUT_DIR / uploaded.name
        dst.write_bytes(uploaded.getvalue())
        st.success(f"{uploaded.name} {i18n.t('upload_mp4')}")
        st.rerun()

    st.divider()

    with st.expander(f"⚙️ {i18n.t('settings')}", expanded=False):
        jump_sec = st.slider(i18n.t("jump_sec"), 1, 30, 5, help=i18n.t("jump_help"))
        ocr_gpu = st.checkbox(i18n.t("ocr_gpu"), value=True)
        ocr_batch_size = st.number_input(i18n.t("ocr_batch_size"), min_value=1, max_value=64, value=16)
        ocr_preprocessing = st.selectbox(
            i18n.t("ocr_preprocessing"),
            options=["none", "grayscale", "threshold", "adaptive", "denoise"],
            format_func=lambda x: {
                "none": i18n.t("pre_none"),
                "grayscale": i18n.t("pre_grayscale"),
                "threshold": i18n.t("pre_threshold"),
                "adaptive": i18n.t("pre_adaptive"),
                "denoise": i18n.t("pre_denoise"),
            }.get(x, x),
        )
        vlm_models = st.multiselect(
            i18n.t("vlm_models"),
            options=[c.name for c in DEFAULT_MODELS],
            default=[DEFAULT_MODELS[0].name],
        )
        vlm_strategy = st.selectbox(
            i18n.t("vlm_strategy"),
            options=["single", "majority", "consensus", "any"],
            format_func=lambda x: {
                "single": i18n.t("strat_single"),
                "majority": i18n.t("strat_majority"),
                "consensus": i18n.t("strat_consensus"),
                "any": i18n.t("strat_any"),
            }.get(x, x),
        )

    st.divider()
    st.caption("OpenProctor v0.1")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

# Show persistent state messages
if st.session_state.pop("batch_stopped_msg", False):
    st.info(i18n.t("batch_stopped"))
if st.session_state.pop("batch_stopping_msg", False):
    st.info("⏹ Stopping after current video finishes ...")
completed_n = st.session_state.pop("batch_completed_msg", False)
if completed_n is not False:
    st.success(i18n.t("batch_completed", n=completed_n))

col_title, col_refresh = st.columns([0.8, 0.2])
with col_title:
    st.title(f"📂 {i18n.t('videos_title')}")
with col_refresh:
    st.write("")
    st.write("")
    if st.button(f"🔄 {i18n.t('refresh')}", use_container_width=True):
        st.rerun()

videos = _scan_videos()

if not videos:
    st.info(i18n.t("no_videos"))
    st.stop()

# --- Table ---
cols = st.columns([3, 1, 1.5, 1.5])
cols[0].markdown(f"**{i18n.t('video')}**")
cols[1].markdown(f"**{i18n.t('size')}**")
cols[2].markdown(f"**{i18n.t('status')}**")
cols[3].markdown(f"**{i18n.t('action')}**")
st.markdown("---")

for v in videos:
    c1, c2, c3, c4 = st.columns([3, 1, 1.5, 1.5])
    c1.markdown(f"`{v['name']}`")
    c2.markdown(f"{v['size_mb']:.0f} {i18n.t('mb')}")

    if v["completed"]:
        c3.markdown(f"<span class='badge-ok'>✅ {i18n.t('completed')}</span>", unsafe_allow_html=True)
        if c4.button(f"📄 {i18n.t('view')}", key=f"view_{v['name']}"):
            st.session_state.selected_video = v["name"]
            st.rerun()
    else:
        c3.markdown(f"<span class='badge-pending'>⏳ {i18n.t('pending')}</span>", unsafe_allow_html=True)
        c4.markdown("—")

st.markdown("---")

# --- Batch controls ---
pending = [v for v in videos if not v["completed"]]
total = len(videos)
done_via_reports = total - len(pending)

if st.session_state.batch_running:
    cols = st.columns([3, 1, 1])
    if st.session_state.batch_stop:
        cols[0].markdown(f"**⏹ {i18n.t('batch_stopping')}**")
        cols[1].button(f"⏹ {i18n.t('stop')}", type="secondary", use_container_width=True, disabled=True)
    else:
        cols[0].markdown(
            f"**⏳ {i18n.t('batch_in_progress', done=st.session_state.batch_done, total=st.session_state.batch_total)}**"
        )
        if cols[1].button(f"⏹ {i18n.t('stop')}", type="secondary", use_container_width=True):
            st.session_state.batch_stop = True
else:
    cols = st.columns([3, 1, 1])
    cols[0].markdown(f"**{i18n.t('processed_count', done=done_via_reports, total=total)}**")
    if len(pending) == 0 and done_via_reports > 0:
        cols[1].success(i18n.t("all_processed"))
    elif cols[1].button(
        f"▶ {i18n.t('process_pending', n=len(pending))}",
        type="primary",
        use_container_width=True,
        disabled=len(pending) == 0,
    ):
        st.session_state.batch_running = True
        st.session_state.batch_queue = [v for v in videos if not v["completed"]]
        st.session_state.batch_done = done_via_reports
        st.session_state.batch_total = total
        st.session_state.batch_stop = False
        _save_batch_state()
        st.rerun()

st.markdown("---")

# Safety reset: if batch is running but queue is empty, force stop
if st.session_state.batch_running and not st.session_state.batch_queue:
    _clear_batch_state()
    st.session_state.batch_running = False
    st.session_state.batch_stop = False

# ---------------------------------------------------------------------------
# Process ONE video per rerun
# ---------------------------------------------------------------------------
if st.session_state.batch_running and st.session_state.batch_queue:

    if st.session_state.batch_stop:
        _clear_batch_state()
        st.session_state.batch_running = False
        st.session_state.batch_queue.clear()
        st.session_state.batch_stopped_msg = True
        st.rerun()

    next_v = st.session_state.batch_queue.pop(0)
    video = next_v["path"]
    name = next_v["name"]

    selected_configs = [c for c in DEFAULT_MODELS if c.name in vlm_models]
    pipeline = Pipeline(
        video_path=video,
        jump_sec=jump_sec,
        ocr_gpu=ocr_gpu,
        ocr_batch_size=ocr_batch_size,
        ocr_preprocessing=ocr_preprocessing,
        vlm_model=selected_configs,
        vlm_strategy=vlm_strategy,
    )

    current_idx = st.session_state.batch_done
    total_b = st.session_state.batch_total

    with st.status(
        f"({current_idx + 1}/{total_b}) {name}",
        expanded=True,
    ) as status:

        video_pbar = st.progress(0, text="—")
        overall_pbar = st.progress(0, text="—")
        icons = {"extraction": "🎬", "ocr": "🔍", "vlm": "🧠"}
        phase_keys = {
            "extraction": "extraction",
            "ocr": "ocr",
            "vlm": "vlm",
        }

        def make_cb(status, video_pbar, overall_pbar, i18n, current_idx, total_b):
            def cb(phase, pct, msg):
                if st.session_state.get("batch_stop", False):
                    raise StopRequested()
                icon = icons.get(phase, "")
                status.update(label=f"({current_idx + 1}/{total_b}) {name}  —  {icon} {msg}")
                video_pbar.progress(pct, text=f"{icon} {i18n.t(phase_keys.get(phase, phase))} — {pct*100:.0f}%")
                overall_pbar.progress(
                    min((current_idx + pct) / total_b, 1.0) if total_b else 0,
                    text=i18n.t("general_progress", done=current_idx, total=total_b),
                )
            return cb

        overall_pbar.progress(
            min(current_idx / total_b, 1.0) if total_b else 0,
            text=i18n.t("general_progress", done=current_idx, total=total_b),
        )
        cb = make_cb(status, video_pbar, overall_pbar, i18n, current_idx, total_b)

        video_ok = False
        try:
            report = pipeline.run(progress=cb)
            n_inf = report.get("summary", {}).get("infractions_confirmed", 0)
            status.update(
                label=f"✅ ({current_idx + 1}/{total_b}) {name} — {n_inf} {i18n.t('infractions_found')}",
                state="complete",
            )
            video_ok = True
        except StopRequested:
            st.session_state.batch_queue.insert(0, next_v)
            _clear_batch_state()
            st.session_state.batch_running = False
            st.session_state.batch_queue.clear()
            st.session_state.batch_stopped_msg = True
            st.rerun()
        except Exception as e:
            status.update(
                label=f"❌ ({current_idx + 1}/{total_b}) {name} — Error: {e}",
                state="error",
            )
            if not (REPORTS_DIR / f"{video.stem}_report.json").exists():
                vid_key = str(video)
                retries = st.session_state.batch_retries
                if retries.get(vid_key, 0) < 2:
                    retries[vid_key] = retries.get(vid_key, 0) + 1
                    st.session_state.batch_queue.insert(0, next_v)
                else:
                    st.warning(f"Skipping {name} after 2 failed attempts")
                    retries.pop(vid_key, None)
                    video_ok = True
            else:
                video_ok = True

    if video_ok:
        st.session_state.batch_done += 1
        st.session_state.batch_processed.append(name)
        st.session_state.batch_retries.pop(str(video), None)
        final_done = st.session_state.batch_done
        overall_pbar.progress(
            min(final_done / total_b, 1.0) if total_b else 0,
            text=i18n.t("general_progress", done=final_done, total=total_b),
        )
        video_pbar.progress(1.0, text="✅ Complete")
    _save_batch_state()

    if not st.session_state.batch_queue:
        _clear_batch_state()
        n = st.session_state.batch_done
        st.session_state.batch_running = False
        st.session_state.batch_completed_msg = n
        st.rerun()
    else:
        st.rerun()

# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------
if st.session_state.selected_video:
    sel = st.session_state.selected_video
    stem = Path(sel).stem
    report_path = REPORTS_DIR / f"{stem}_report.json"

    if report_path.exists():
        report = _load_report(report_path)
        summary = report.get("summary", {})
        infractions = report.get("infractions", [])

        st.divider()
        col_back, col_title = st.columns([0.1, 0.9])
        with col_back:
            if st.button(i18n.t("back")):
                st.session_state.selected_video = None
                st.rerun()
        with col_title:
            st.subheader(f"📄 {sel}")

        mc = st.columns(4)
        mc[0].metric(f"🎬 {i18n.t('frames_extracted')}", summary.get("total_frames_extracted", 0))
        mc[1].metric(f"🔍 {i18n.t('suspects_ocr')}", summary.get("suspects_found", 0))
        mc[2].metric(f"🧠 {i18n.t('analyzed_vlm')}", summary.get("suspects_found", 0))
        mc[3].metric(f"🛑 {i18n.t('infractions')}", summary.get("infractions_confirmed", 0))

        st.divider()

        # --- Collect frames ---
        interim_dir = _find_frames_dir(INTERIM_BASE, stem)
        suspects_dir = _find_frames_dir(SUSPECTS_BASE, stem)

        all_frames: list[Path] = []
        if interim_dir:
            all_frames.extend(sorted(interim_dir.glob("*.jpg")))
        suspect_frames: list[Path] = []
        findings_map: dict[str, dict] = {}
        if suspects_dir:
            suspect_frames = sorted(suspects_dir.glob("*.jpg"))
            all_frames.extend(suspect_frames)
            f_json = suspects_dir / "findings.json"
            if f_json.exists():
                for entry in json.loads(f_json.read_text()):
                    findings_map[Path(entry["file"]).name] = entry
        all_frames.sort(key=lambda p: p.name)

        # --- All Extracted Frames ---
        if all_frames:
            with st.expander(f"🎬 {i18n.t('frames_extracted')} ({len(all_frames)})", expanded=False):
                _show_frame_grid(all_frames, f"all_{stem}", findings_map=findings_map)

        # --- OCR Suspects ---
        if suspect_frames:
            with st.expander(f"🔍 {i18n.t('suspects_ocr')} ({len(suspect_frames)})", expanded=True):
                _show_frame_grid(suspect_frames, f"suspect_{stem}", findings_map=findings_map)

        # --- Infractions ---
        if infractions:
            st.subheader(f"🛑 {len(infractions)} {i18n.t('infractions_found')}")
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
                            st.caption("(image not available)")
                    with mc2:
                        st.markdown(f"### #{i} — ⏱ `{ts}`")
                        if kw:
                            st.markdown(
                                f"<span class='keyword-badge'>{kw}</span>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"**🧠 {i18n.t('ai_reason')}:**  \n{reason}")
                        if ocr_text:
                            with st.expander(f"📝 {i18n.t('ocr_text')}"):
                                st.text(ocr_text[:500])
        else:
            st.info(f"✅ {i18n.t('no_infractions')}")

    else:
        st.warning(i18n.t("report_not_found", path=report_path))
        if st.button(i18n.t("back")):
            st.session_state.selected_video = None
            st.rerun()
