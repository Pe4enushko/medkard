from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any
from dotenv import load_dotenv

load_dotenv()

class OneCClient:
    appointments_url_env = "ALENKA_ONE_C_APPOINTMENTS_URL"
    login_env = "ALENKA_ONE_C_LOGIN"
    password_env = "ALENKA_ONE_C_PASSWORD"
    timeout_seconds_env = "ALENKA_ONE_C_TIMEOUT_SECONDS"

    def __init__(
        self,
        url: str,
        login: str,
        password: str,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.url = url
        self.login = login
        self.password = password
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OneCClient":
        return cls(
            url=os.environ[cls.appointments_url_env],
            login=os.environ[cls.login_env],
            password=os.environ[cls.password_env],
            timeout_seconds=float(os.environ.get(cls.timeout_seconds_env, 15)),
        )

    def fetch_json_for_period(self, datebegin: str, dateend: str) -> Any:
        """Fetch raw JSON from 1C for a given date period and return it as-is.

        Args:
            datebegin: Period start date in format expected by 1C (e.g. DD.MM.YYYY).
            dateend: Period end date in format expected by 1C (e.g. DD.MM.YYYY).
        """
        if not self.url or self.url.startswith("<"):
            raise ValueError(f"Set real {self.appointments_url_env} in environment")
        if not self.login or not self.password:
            raise ValueError(f"{self.login_env} and {self.password_env} must be set")

        token = base64.b64encode(f"{self.login}:{self.password}".encode("utf-8")).decode("ascii")
        query_params = urllib.parse.urlencode({"datebegin": datebegin, "dateend": dateend})
        separator = "&" if "?" in self.url else "?"
        request_url = f"{self.url}{separator}{query_params}"

        request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "Authorization": f"Basic {token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch appointments from 1C: {exc}") from exc

    def fetch_json_for_today(self) -> Any:
        """Fetch raw JSON from 1C for today's date and return it as-is."""
        current_day = datetime.now().strftime("%d.%m.%Y")
        return self.fetch_json_for_period(datebegin=current_day, dateend=current_day)


class AlenkaOneCClient(OneCClient):
    pass


class MdsOneCClient(OneCClient):
    appointments_url_env = "MDS_ONE_C_APPOINTMENTS_URL"
    login_env = "MDS_ONE_C_LOGIN"
    password_env = "MDS_ONE_C_PASSWORD"
    timeout_seconds_env = "MDS_ONE_C_TIMEOUT_SECONDS"
