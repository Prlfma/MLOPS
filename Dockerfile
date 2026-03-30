# Етап 1: Builder
FROM python:3.14-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Етап 2: Фінальний образ
FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/* && rm -rf /wheels
COPY src/ src/
COPY dvc.yaml dvc.lock ./
CMD ["dvc", "repro"]
