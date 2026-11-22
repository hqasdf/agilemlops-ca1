FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir "mlflow==3.17.2" "PyMySQL==1.1.1" "boto3==1.40.61" "cryptography==45.0.7"
COPY start_mlflow.py /app/start_mlflow.py
COPY certs/rds-bundle.pem /app/certs/rds-bundle.pem
EXPOSE 5000
CMD ["python", "/app/start_mlflow.py"]
