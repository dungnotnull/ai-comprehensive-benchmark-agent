FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/models

RUN groupadd -r agentuser && useradd -r -g agentuser agentuser
RUN chown -R agentuser:agentuser /app
USER agentuser

EXPOSE 8022

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8022/health || exit 1

CMD ["python", "-m", "agent.main", "start-proxy", "--proxy-port", "8022"]
