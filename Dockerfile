# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    # Add font packages
    fonts-liberation \
    fontconfig \
    # Install ImageMagick
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create only static directories (remove media creation)
RUN mkdir -p static staticfiles

# Copy project files
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Copy and set ImageMagick policy
COPY policy.xml /etc/ImageMagick-6/policy.xml

# Expose ports
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=youtube_narration.settings
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=django-db

# Run migrations
RUN python manage.py migrate

# Create entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command (can be overridden in docker-compose.yml)
CMD ["gunicorn", "youtube_narration.wsgi:application", "--bind", "0.0.0.0:8000"]