FROM python:3.10-slim

# Needed for numpy & pillow
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY best_model.h5 .

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
