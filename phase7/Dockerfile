FROM python:3.11-slim
WORKDIR /app
COPY phase5/requirements-phase5.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
COPY artifacts /app/artifacts
COPY phase5/api /app/phase5/api
ENV PROJECT_ROOT=/app
EXPOSE 5000
CMD ["python", "/app/phase5/api/app.py", "--host", "0.0.0.0", "--port", "5000"]
