FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serial_writer ./serial_writer

RUN useradd --create-home --uid 1000 app && mkdir /data && chown app:app /data
USER app

WORKDIR /data
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)"

CMD ["python", "-m", "serial_writer.web", "--host", "0.0.0.0", "--port", "8000"]
