from __future__ import annotations

import shutil
import uuid
from pathlib import Path, PurePosixPath

import streamlit as st

from agents.butterbase_client import ButterbaseClient, extract_row_id
from agents.local_pipeline import run_pipeline


UPLOAD_ROOT = Path("uploads")
REPORT_ROOT = Path("reports")


def safe_upload_path(name: str) -> Path | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    clean_parts = [part for part in path.parts if part not in ("", ".")]
    if not clean_parts:
        return None
    return Path(*clean_parts)


def save_uploaded_folder(uploaded_files: list) -> Path:
    run_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOAD_ROOT / run_id
    upload_dir.mkdir(parents=True, exist_ok=False)

    for uploaded_file in uploaded_files:
        relative_path = safe_upload_path(uploaded_file.name)
        if relative_path is None:
            continue
        destination = upload_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(uploaded_file.getbuffer())

    return upload_dir


def render_summary(result: dict) -> None:
    summary = result["compatibility"]["summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Operator uses", summary["total_operator_uses"])
    col2.metric("Unique ops", summary["unique_operator_count"])
    col3.metric("Conditional", summary["conditional_unique_count"])
    col4.metric("Unsupported", summary["unsupported_unique_count"])

    st.subheader("Compatibility")
    st.dataframe(result["compatibility"]["compatibility_table"], use_container_width=True)

    report_path = Path(result["report_path"])
    report_text = report_path.read_text(encoding="utf-8")
    st.download_button(
        "Download report",
        data=report_text,
        file_name=report_path.name,
        mime="text/markdown",
    )

    with st.expander("Report preview", expanded=True):
        st.markdown(report_text)


def persist_analysis(source: str, analysis_dir: Path, report_path: Path) -> dict:
    butterbase = ButterbaseClient.from_env()
    job_id = None
    if butterbase:
        created = butterbase.create_job(source=source, model_input_folder=str(analysis_dir))
        job_id = extract_row_id(created)

    try:
        result = run_pipeline(analysis_dir, report_path)
        if butterbase and job_id:
            butterbase.complete_job(job_id, result)
        result["butterbase_job_id"] = job_id
        return result
    except Exception as exc:
        if butterbase and job_id:
            butterbase.fail_job(job_id, str(exc))
        raise


def main() -> None:
    st.set_page_config(page_title="Model Portability Analyzer", layout="wide")
    st.title("Model Portability Analyzer")

    uploaded_files = st.file_uploader(
        "Upload a model source folder",
        accept_multiple_files="directory",
    )

    use_existing = st.checkbox("Use existing sample folder", value=not uploaded_files)

    if use_existing:
        model_dir = Path("model_input/Transfer-Model_original")
        st.caption(str(model_dir))
    elif uploaded_files:
        st.caption(f"{len(uploaded_files)} files selected")
    else:
        st.info("Select a folder containing the model source files.")
        return

    if st.button("Analyze model folder", type="primary"):
        with st.spinner("Scanning model code and generating report..."):
            if use_existing:
                analysis_dir = model_dir
                run_id = "sample"
            else:
                analysis_dir = save_uploaded_folder(uploaded_files)
                run_id = analysis_dir.name

            report_path = REPORT_ROOT / f"{run_id}_vitis_ai_compatibility.md"
            source = "sample" if use_existing else "streamlit-upload"
            result = persist_analysis(source, analysis_dir, report_path)
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        render_summary(st.session_state["last_result"])

    with st.sidebar:
        st.header("Butterbase")
        butterbase = ButterbaseClient.from_env()
        if butterbase:
            st.success("Configured")
            if st.button("Apply schema"):
                with st.spinner("Applying Butterbase schema..."):
                    butterbase.apply_schema()
                st.success("Schema applied")
            if "last_result" in st.session_state and st.session_state["last_result"].get("butterbase_job_id"):
                st.caption(f"Last job: {st.session_state['last_result']['butterbase_job_id']}")
        else:
            st.info("Set BUTTERBASE_APP_ID and BUTTERBASE_API_KEY to persist jobs.")

        st.header("Storage")
        if st.button("Clear uploaded folders"):
            shutil.rmtree(UPLOAD_ROOT, ignore_errors=True)
            st.session_state.pop("last_result", None)
            st.rerun()


if __name__ == "__main__":
    main()
