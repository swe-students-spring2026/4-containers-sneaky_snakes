FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --deploy --system

COPY . .

CMD ["python", "app.py"]
