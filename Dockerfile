FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/state/data \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only application inputs: local environments, secrets and private notes stay out.
COPY src/ ./src/
COPY ui/ ./ui/
COPY data/ /opt/seed-data/

# B's code resolves warehouse/reports relative to /app; both services share their targets.
RUN mkdir -p /state /app/models \
    && ln -s /state/warehouse.db /app/warehouse.db \
    && ln -s /state/data /app/data \
    && ln -s /state/reports /app/reports

EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
