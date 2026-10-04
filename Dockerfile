FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Non-root; weights and logs live on volumes so they survive container replacement
RUN useradd -m xengine && mkdir -p /app/models /app/logs && chown -R xengine /app
USER xengine
VOLUME ["/app/models", "/app/logs"]
EXPOSE 8765
HEALTHCHECK --interval=2m --timeout=10s --start-period=3m --retries=2 CMD python healthcheck.py 600
# Inside a container 127.0.0.1 is unreachable from the host: bind 0.0.0.0 AND require MCP_AUTH_TOKEN
# (run_sse refuses a non-loopback bind without it). Credentials come from --env-file, never the image.
CMD ["python", "launch.py", "--forever", "--http", "--host", "0.0.0.0", "--port", "8765"]
