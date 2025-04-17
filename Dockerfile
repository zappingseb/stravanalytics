FROM debian:bullseye
WORKDIR /app

# Install Python packages from apt
RUN  apt-get update && apt-get install -y python3.10 python3-pip python3-yaml python3-numpy && rm -rf /var/lib/apt/lists/*
RUN  apt-get update && apt-get install -y python3-pandas python3-requests \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install remaining pip packages
COPY requirements.txt .
RUN pip3 install --target=/usr/lib/python3/site-packages -r requirements.txt

COPY . .
# -- PRODUCTION MODE
# ENV PYTHONPATH=/usr/lib/python3/site-packages:/usr/lib/python3/dist-packages:/app

# EXPOSE 5000
# CMD ["python3", "app.py"]
# -- EOF PRODUCTION MODE

ENV PYTHONPATH=/usr/lib/python3/site-packages:/usr/lib/python3/dist-packages:/app
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=1

EXPOSE 5000
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]