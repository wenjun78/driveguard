# Console image (alternative to the Render Python runtime): docker build -t driveguard-console . && docker run -p 8000:8000 driveguard-console
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[app]"
COPY web ./web
COPY app_snapshot ./app_snapshot
ENV MLFLOW_DISABLE_AGENT_HINT=1 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn driveguard.app.server:app --host 0.0.0.0 --port ${PORT}"]
