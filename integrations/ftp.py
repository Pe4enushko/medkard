"""FTP upload helper for report scripts."""

from __future__ import annotations

import ftplib
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def load_creds(path: str) -> dict[str, str]:
    """Parse a key=value credentials file. Raises ValueError on missing keys."""
    creds: dict[str, str] = {}
    required = {"ip", "port", "username", "password"}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        creds[key.strip()] = value.strip()
    missing = required - creds.keys()
    if missing:
        raise ValueError(f"Credentials file missing keys: {', '.join(sorted(missing))}")
    return creds


def _makedirs(ftp: ftplib.FTP, path: str) -> None:
    parts = [p for p in path.split("/") if p]
    current = ""
    for part in parts:
        current += f"/{part}"
        try:
            ftp.mkd(current)
        except ftplib.error_perm as e:
            if "550" not in str(e):
                raise


def upload(local_path: Path, filename: str, creds: dict[str, str]) -> None:
    """Upload *local_path* to /<year>/<month>/<filename> on the FTP server."""
    now = datetime.now()
    remote_dir = f"/{now.year}/{now.month:02d}"
    remote_path = f"{remote_dir}/{filename}"

    log.info("Connecting to FTP %s:%s", creds["ip"], creds["port"])
    with ftplib.FTP() as ftp:
        ftp.connect(host=creds["ip"], port=int(creds["port"]))
        ftp.login(user=creds["username"], passwd=creds["password"])
        ftp.set_pasv(True)
        _makedirs(ftp, remote_dir)
        ftp.cwd(remote_dir)
        with local_path.open("rb") as f:
            ftp.storbinary(f"STOR {filename}", f)

    log.info("Uploaded → ftp://%s%s", creds["ip"], remote_path)
