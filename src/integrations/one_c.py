from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any
from dotenv import load_dotenv

load_dotenv()

class OneCClient:
    appointments_url_env = "ALENKA_ONE_C_APPOINTMENTS_URL"
    login_env = "ALENKA_ONE_C_LOGIN"
    password_env = "ALENKA_ONE_C_PASSWORD"
    timeout_seconds_env = "ALENKA_ONE_C_TIMEOUT_SECONDS"
    requires_password = True

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
            password=os.environ[cls.password_env] if cls.requires_password else os.environ.get(cls.password_env, ""),
            timeout_seconds=float(os.environ.get(cls.timeout_seconds_env, 15)),
        )

    def _check_credentials(self) -> None:
        if not self.url or self.url.startswith("<"):
            raise ValueError(f"Set real {self.appointments_url_env} in environment")
        if not self.login:
            raise ValueError(f"{self.login_env} must be set")
        if self.requires_password and not self.password:
            raise ValueError(f"{self.password_env} must be set")

    def _basic_auth_header(self) -> str:
        token = base64.b64encode(f"{self.login}:{self.password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _send(self, request: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch appointments from 1C: {exc}") from exc

    def fetch_json_for_period(self, datebegin: str, dateend: str) -> Any:
        """Fetch raw JSON from 1C for a given date period and return it as-is.

        Args:
            datebegin: Period start date in format expected by 1C (e.g. DD.MM.YYYY).
            dateend: Period end date in format expected by 1C (e.g. DD.MM.YYYY).
        """
        self._check_credentials()
        query_params = urllib.parse.urlencode({"datebegin": datebegin, "dateend": dateend})
        separator = "&" if "?" in self.url else "?"
        request_url = f"{self.url}{separator}{query_params}"

        request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "Authorization": self._basic_auth_header()},
            method="GET",
        )
        return self._send(request)

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
    requires_password = False

    def fetch_json_for_period(self, datebegin: str, dateend: str) -> Any:
        """MDS 1C accepts one day per request: POST {"date": "YYYY-MM-DD"}.

        A multi-day period becomes one POST per day; day payloads are merged
        into a flat visit list (a single-visit dict payload is wrapped).
        """
        self._check_credentials()
        begin = datetime.strptime(datebegin, "%d.%m.%Y")
        end = datetime.strptime(dateend, "%d.%m.%Y")

        visits: list[Any] = []
        day = begin
        while day <= end:
            payload = self._fetch_json_for_date(day.strftime("%Y-%m-%d"))
            if isinstance(payload, list):
                visits.extend(payload)
            elif payload:
                visits.append(payload)
            day += timedelta(days=1)
        return visits

    def _fetch_json_for_date(self, date: str) -> Any:
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"date": date}).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self._basic_auth_header(),
            },
            method="POST",
        )
        return self._send(request)
