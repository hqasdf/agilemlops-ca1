FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db \
    MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts \
    MLFLOW_ALLOWED_HOSTS=localhost:*,127.0.0.1:*

RUN pip install --no-cache-dir "mlflow==3.17.2" \
    && mkdir -p /mlflow/artifacts
WORKDIR /mlflow
EXPOSE 5000

CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port ${PORT} --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} --allowed-hosts ${MLFLOW_ALLOWED_HOSTS}"]
