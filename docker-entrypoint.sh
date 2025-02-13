#!/bin/bash

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Start Celery in the background if the command contains celery
if [[ "$*" == *"celery"* ]]; then
    echo "Starting Celery..."
    exec "$@"
elif [[ "$*" == *"beat"* ]]; then
    echo "Starting Celery Beat..."
    exec "$@"
else
    # Start the main application
    echo "Starting Django application..."
    exec "$@"
fi 