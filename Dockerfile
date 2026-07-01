FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements-api.txt .
RUN uv pip install --system --no-cache -r requirements-api.txt

COPY src ./src
COPY resources/manifest.csv ./resources/manifest.csv

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
