FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
COPY constraints.container.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -c constraints.container.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.server.api:app", "--host", "0.0.0.0", "--port", "8000"]
