from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ANALYSIS_JOBS_SCHEMA = {
    "schema": {
        "tables": {
            "analysis_jobs": {
                "columns": {
                    "id": {"type": "uuid", "primary": True, "default": "gen_random_uuid()"},
                    "source": {"type": "text", "nullable": False},
                    "model_input_folder": {"type": "text", "nullable": False},
                    "target": {"type": "text", "default": "'amd-vitis-ai'"},
                    "status": {"type": "text", "nullable": False, "default": "'queued'"},
                    "report_path": {"type": "text"},
                    "report_markdown": {"type": "text"},
                    "total_operator_uses": {"type": "integer", "default": "0"},
                    "unique_operator_count": {"type": "integer", "default": "0"},
                    "supported_unique_count": {"type": "integer", "default": "0"},
                    "conditional_unique_count": {"type": "integer", "default": "0"},
                    "unsupported_unique_count": {"type": "integer", "default": "0"},
                    "result_json": {"type": "jsonb"},
                    "error": {"type": "text"},
                    "created_at": {"type": "timestamptz", "default": "now()"},
                    "updated_at": {"type": "timestamptz", "default": "now()"},
                },
                "indexes": {
                    "analysis_jobs_status_idx": {"columns": ["status"]},
                    "analysis_jobs_created_at_idx": {"columns": ["created_at"]},
                },
            }
        }
    },
    "dry_run": False,
}


@dataclass(frozen=True)
class ButterbaseConfig:
    app_id: str
    api_key: str
    api_url: str = "https://api.butterbase.ai"


class ButterbaseClient:
    def __init__(self, config: ButterbaseConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> ButterbaseClient | None:
        app_id = os.getenv("BUTTERBASE_APP_ID")
        api_key = os.getenv("BUTTERBASE_API_KEY")
        api_url = os.getenv("BUTTERBASE_API_URL", "https://api.butterbase.ai")
        if not app_id or not api_key:
            return None
        return cls(ButterbaseConfig(app_id=app_id, api_key=api_key, api_url=api_url.rstrip("/")))

    @property
    def configured(self) -> bool:
        return bool(self.config.app_id and self.config.api_key)

    def apply_schema(self) -> dict[str, Any]:
        return self._request("POST", f"/v1/{self.config.app_id}/schema/apply", json=ANALYSIS_JOBS_SCHEMA)

    def create_job(self, *, source: str, model_input_folder: str, target: str = "amd-vitis-ai") -> dict[str, Any]:
        payload = {
            "source": source,
            "model_input_folder": model_input_folder,
            "target": target,
            "status": "running",
        }
        return self._request("POST", f"/v1/{self.config.app_id}/analysis_jobs", json=payload)

    def complete_job(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        summary = result["compatibility"]["summary"]
        report_path = Path(result["report_path"])
        report_markdown = report_path.read_text(encoding="utf-8") if report_path.exists() else None
        payload = {
            "status": "complete",
            "report_path": result["report_path"],
            "report_markdown": report_markdown,
            "total_operator_uses": summary["total_operator_uses"],
            "unique_operator_count": summary["unique_operator_count"],
            "supported_unique_count": summary["supported_unique_count"],
            "conditional_unique_count": summary["conditional_unique_count"],
            "unsupported_unique_count": summary["unsupported_unique_count"],
            "result_json": result,
        }
        return self._request("PATCH", f"/v1/{self.config.app_id}/analysis_jobs/{job_id}", json=payload)

    def fail_job(self, job_id: str, error: str) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/v1/{self.config.app_id}/analysis_jobs/{job_id}",
            json={"status": "failed", "error": error},
        )

    def list_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._request(
            "GET",
            f"/v1/{self.config.app_id}/analysis_jobs",
            params={"order": "created_at.desc", "limit": limit},
        )
        return rows if isinstance(rows, list) else rows.get("data", rows)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = kwargs.pop("headers", {})
        headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )
        response = requests.request(
            method,
            f"{self.config.api_url}{path}",
            headers=headers,
            timeout=30,
            **kwargs,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def extract_row_id(row: Any) -> str | None:
    if isinstance(row, dict):
        if "id" in row:
            return str(row["id"])
        data = row.get("data")
        if isinstance(data, dict) and "id" in data:
            return str(data["id"])
        if isinstance(data, list) and data and isinstance(data[0], dict) and "id" in data[0]:
            return str(data[0]["id"])
    if isinstance(row, list) and row and isinstance(row[0], dict) and "id" in row[0]:
        return str(row[0]["id"])
    return None
