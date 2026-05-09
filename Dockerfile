FROM debian:bullseye
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --target=/usr/lib/python3/site-packages -r requirements.txt

COPY . .

ENV PYTHONPATH=/usr/lib/python3/site-packages:/usr/lib/python3/dist-packages:/app

EXPOSE 8080
CMD ["python3", "-m", "gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
