.PHONY: build up down restart logs shell migrate static test clean help stop

# Variables
IMAGE_NAME = youtube-narration
CONTAINER_NAME = youtube-narration-app
PORT_DOCKER = 8001
PORT_SYSTEM = 8000
# Colors for terminal output
GREEN = \033[0;32m
NC = \033[0m # No Color

help:
	@echo "Available commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all containers in detached mode"
	@echo "  make stop       - Stop running containers without removing them"
	@echo "  make down       - Stop and remove all containers"
	@echo "  make restart    - Restart all containers"
	@echo "  make logs       - View logs from all containers"
	@echo "  make shell      - Open a shell in the web container"
	@echo "  make migrate    - Run Django migrations"
	@echo "  make static     - Collect static files"
	@echo "  make test       - Run Django tests"
	@echo "  make clean      - Remove all containers, volumes, and images"

build:
	docker-compose build

up:
	docker-compose up -d

stop:
	docker-compose stop

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec web python manage.py shell

migrate:
	docker-compose exec web python manage.py migrate

static:
	docker-compose exec web python manage.py collectstatic --noinput

test:
	docker-compose exec web python manage.py test

clean:
	docker-compose down -v --rmi all

# Development specific commands
dev-build:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

dev-up:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

dev-down:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

# Production specific commands
prod-build:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-down:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

# Celery specific commands
celery-logs:
	docker-compose logs -f celery

celery-restart:
	docker-compose restart celery celery-beat

# Redis specific commands
redis-cli:
	docker-compose exec redis redis-cli

redis-monitor:
	docker-compose exec redis redis-cli monitor

# Database backup and restore
backup:
	docker-compose exec web python manage.py dumpdata > backup.json

restore:
	docker-compose exec web python manage.py loaddata backup.json

# Add a command to watch for changes
watch:
	@echo "Watching for changes..."
	watchmedo auto-restart \
		--directory=./ \
		--pattern=*.py \
		--recursive \
		-- python manage.py runserver 0.0.0.0:$(PORT)