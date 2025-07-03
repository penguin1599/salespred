# Use a slim official Python image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy requirements first (better layer caching) and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Train the model during build (optional – reproducible image)
RUN python train.py

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]