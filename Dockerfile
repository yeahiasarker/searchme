# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create directory for index storage
RUN mkdir -p /root/.file_search_index

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=ollama

# Default command
CMD ["searchme"] 