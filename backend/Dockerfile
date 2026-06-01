# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (build-essential for compiling standard dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root build context
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files into container workdir
COPY backend/ .

# Expose port for the FastAPI WebSocket/HTTP endpoints
EXPOSE 8000

# Start server using the uvicorn launcher script
CMD ["python", "uvicorn_run.py"]
