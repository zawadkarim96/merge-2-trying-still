"""File upload management utilities."""
from __future__ import annotations

import mimetypes
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import AppConfig


@dataclass
class UploadManager:
    config: AppConfig

    def sanitize_filename(self, filename: str) -> str:
        name, ext = os.path.splitext(filename)
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in name).strip("._")
        safe_name = safe_name or "document"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{safe_name}_{timestamp}{ext.lower()}"

    def save(self, uploaded_file, subdir: str) -> Optional[str]:
        if not uploaded_file:
            return None
        if not self._is_allowed(uploaded_file):
            st.error("Unsupported file type uploaded.")
            return None
        target_dir = self.config.data_dir / "uploads" / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.sanitize_filename(uploaded_file.name)
        destination = target_dir / safe_name
        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())
        self._scan_file(destination)
        return str(destination.relative_to(self.config.data_dir))

    def enforce_retention(self) -> None:
        if not self.config.upload_retention:
            return
        cutoff = datetime.now() - self.config.upload_retention
        uploads_root = self.config.data_dir / "uploads"
        for path in uploads_root.rglob("*"):
            if path.is_file():
                if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                    path.unlink(missing_ok=True)

    def metadata(self, relative_path: str) -> Optional[dict]:
        if not relative_path:
            return None
        file_path = (self.config.data_dir / relative_path).resolve()
        if not file_path.exists():
            return None
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "uploaded": datetime.fromtimestamp(stat.st_mtime),
            "path": file_path,
        }

    def _is_allowed(self, uploaded_file) -> bool:
        mimetype = getattr(uploaded_file, "type", None)
        if mimetype and mimetype in self.config.allowed_mime_types:
            return True
        guessed, _ = mimetypes.guess_type(uploaded_file.name)
        return guessed in self.config.allowed_mime_types

    def _scan_file(self, path: Path) -> None:
        if not self.config.virus_scan_command:
            return
        command = [self.config.virus_scan_command, str(path)]
        try:
            result = subprocess.run(command, capture_output=True, check=False, text=True)
        except OSError as exc:
            st.warning(f"Virus scan failed to start: {exc}")
            return
        if result.returncode != 0:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Uploaded file rejected by virus scanner: {result.stdout or result.stderr}"
            )
