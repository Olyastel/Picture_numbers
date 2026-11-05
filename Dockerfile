FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py .
ENV HOST=0.0.0.0
ENV PORT=10000
ENV YANDEX_OCR_MODEL=handwritten
EXPOSE 10000
CMD ["sh", "-c", "python app.py --host ${HOST} --port ${PORT}"]
