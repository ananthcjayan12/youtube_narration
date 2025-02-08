# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static staticfiles media

# Copy project files
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose ports
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=youtube_narration.settings

# Use gunicorn for production
CMD gunicorn youtube_narration.wsgi:application --bind 0.0.0.0:8000