FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# TODO: Set the application command when the serving API is implemented.
CMD ["python", "scripts/evaluate_model.py"]