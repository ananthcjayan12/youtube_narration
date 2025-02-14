.PHONY: build build-no-cache up stop down restart logs shell migrate static test clean help celery-logs celery-restart redis-cli redis-monitor backup restore watch

# Variables
IMAGE_NAME = youtube-narration
CONTAINER_NAME = youtube-narration-app
PORT_DOCKER = 8001
PORT_SYSTEM = 8000
# Colors for terminal output
GREEN = \033[0;32m
NC = \033[0m # No Color

# Docker Compose command for dev environment
DC = docker-compose -f docker-compose.dev.yml

help:
	@echo "Available commands (Dev Environment):"
	@echo "  make build           - Build all Docker images"
	@echo "  make build-no-cache  - Build all Docker images without cache"
	@echo "  make up              - Start all containers in detached mode"
	@echo "  make stop            - Stop running containers"
	@echo "  make down            - Stop and remove all containers"
	@echo "  make restart         - Restart all containers"
	@echo "  make logs            - View logs from all containers"
	@echo "  make shell           - Open a shell in the web container"
	@echo "  make migrate         - Run Django migrations"
	@echo "  make static          - Collect static files"
	@echo "  make test            - Run Django tests"
	@echo "  make clean           - Remove all containers, volumes, and images"
	@echo "  make celery-logs     - View Celery logs"
	@echo "  make celery-restart  - Restart Celery services"
	@echo "  make redis-cli       - Open Redis CLI"
	@echo "  make redis-monitor   - Monitor Redis"
	@echo "  make backup          - Backup the database"
	@echo "  make restore         - Restore the database"
	@echo "  make watch           - Watch for changes and auto-restart server"

build:
	$(DC) build

build-no-cache:
	$(DC) build --no-cache

up:
	$(DC) up -d

stop:
	$(DC) stop

down:
	$(DC) down

restart:
	$(DC) restart

logs:
	$(DC) logs -f

shell:
	$(DC) exec web python manage.py shell

migrate:
	$(DC) exec web python manage.py migrate

static:
	$(DC) exec web python manage.py collectstatic --noinput

test:
	$(DC) exec web python manage.py test

clean:
	$(DC) down -v --rmi all

celery-logs:
	$(DC) logs -f celery

celery-restart:
	$(DC) restart celery celery-beat

redis-cli:
	$(DC) exec redis redis-cli

redis-monitor:
	$(DC) exec redis redis-cli monitor

backup:
	$(DC) exec web python manage.py dumpdata > backup.json

restore:
	$(DC) exec web python manage.py loaddata backup.json

watch:
	@echo "Watching for changes..."
	watchmedo auto-restart \
		--directory=./ \
		--pattern=*.py \
		--recursive \
		-- python manage.py runserver 0.0.0.0:$(PORT)