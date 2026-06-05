from __future__ import annotations

import hmac
import json
import os
import time
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from agents.butterbase_client import ButterbaseClient, extract_row_id
from agents.local_pipeline import run_pipeline


app = FastAPI(title="Model Portability Photon Backend")


class AnalyzeExistingFolderRequest(BaseModel):
    model_input_folder: str = "model_input/Transfer-Model_original"
    report_path: str = "reports/photon_vitis_ai_compatibility.md"


def verify_spectrum_signature(
    body: bytes,
    signature_header: str | None,
    timestamp_header: str | None,
) -> bool:
    secret = os.getenv("SPECTRUM_WEBHOOK_SECRET")
    if not secret:
        return True
    if not signature_header or not timestamp_header:
        return False

    try:
        timestamp = int(timestamp_header)
    except ValueError:
        return False

    if abs(int(time.time()) - timestamp) > 300:
        return False

    expected = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    candidates = [part.strip() for part in signature_header.split(",")]
    for candidate in candidates:
        value = candidate.removeprefix("v0=")
        if hmac.compare_digest(value, expected):
            return True
    return False


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "butterbase_configured": ButterbaseClient.from_env() is not None}


@app.post("/analyze-existing-folder")
async def analyze_existing_folder(payload: AnalyzeExistingFolderRequest) -> dict:
    result = run_and_persist_analysis(
        source="api",
        model_input_folder=Path(payload.model_input_folder),
        report_path=Path(payload.report_path),
    )
    summary = result["compatibility"]["summary"]
    return {
        "status": "complete",
        "report_path": result["report_path"],
        "summary": summary,
        "butterbase_job_id": result.get("butterbase_job_id"),
    }


def run_and_persist_analysis(source: str, model_input_folder: Path, report_path: Path) -> dict:
    butterbase = ButterbaseClient.from_env()
    job_id = None
    if butterbase:
        created = butterbase.create_job(source=source, model_input_folder=str(model_input_folder))
        job_id = extract_row_id(created)

    try:
        result = run_pipeline(model_input_folder, report_path)
        if butterbase and job_id:
            butterbase.complete_job(job_id, result)
        result["butterbase_job_id"] = job_id
        return result
    except Exception as exc:
        if butterbase and job_id:
            butterbase.fail_job(job_id, str(exc))
        raise


@app.post("/spectrum-webhook")
async def spectrum_webhook(
    request: Request,
    x_spectrum_signature: str | None = Header(default=None),
    x_spectrum_timestamp: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    if not verify_spectrum_signature(body, x_spectrum_signature, x_spectrum_timestamp):
        raise HTTPException(status_code=401, detail="Invalid Spectrum signature")

    payload = json.loads(body or b"{}")
    message = payload.get("message", {})
    content = message.get("content", "")
    if isinstance(content, dict):
        content = json.dumps(content)

    if "analyze" in str(content).lower():
        result = run_and_persist_analysis(
            source="photon",
            model_input_folder=Path("model_input/Transfer-Model_original"),
            report_path=Path("reports/photon_vitis_ai_compatibility.md"),
        )
        return {
            "ok": True,
            "action": "analysis_complete",
            "report_path": result["report_path"],
            "summary": result["compatibility"]["summary"],
            "butterbase_job_id": result.get("butterbase_job_id"),
        }

    return {
        "ok": True,
        "action": "ignored",
        "hint": "Send a message containing 'analyze' to trigger the sample model analysis.",
    }
