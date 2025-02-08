.PHONY: build run stop clean shell logs test migrate static help

# Variables
IMAGE_NAME = youtube-narration
CONTAINER_NAME = youtube-narration-app
PORT = 8000

# Colors for terminal output
GREEN = \033[0;32m
NC = \033[0m # No Color

help:
	@echo "Available commands:"
	@echo "${GREEN}make build${NC}      - Build Docker image"
	@echo "${GREEN}make run${NC}        - Run Docker container"
	@echo "${GREEN}make stop${NC}       - Stop Docker container"
	@echo "${GREEN}make restart${NC}    - Restart Docker container"
	@echo "${GREEN}make shell${NC}      - Access container shell"
	@echo "${GREEN}make logs${NC}       - View container logs"
	@echo "${GREEN}make clean${NC}      - Remove container and image"
	@echo "${GREEN}make migrate${NC}    - Run Django migrations"
	@echo "${GREEN}make static${NC}     - Collect static files"
	@echo "${GREEN}make test${NC}       - Run tests"

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Running container..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		-v $(PWD):/app \
		--env-file .env \
		$(IMAGE_NAME)
	@echo "Container started at http://localhost:$(PORT)"

stop:
	@echo "Stopping container..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

restart: stop run

shell:
	@echo "Accessing container shell..."
	docker exec -it $(CONTAINER_NAME) bash

logs:
	@echo "Viewing logs..."
	docker logs -f $(CONTAINER_NAME)

clean: stop
	@echo "Removing image..."
	docker rmi $(IMAGE_NAME) || true

migrate:
	@echo "Running migrations..."
	docker exec -it $(CONTAINER_NAME) python manage.py migrate

static:
	@echo "Collecting static files..."
	docker exec -it $(CONTAINER_NAME) python manage.py collectstatic --noinput

test:
	@echo "Running tests..."
	docker exec -it $(CONTAINER_NAME) python manage.py test

dev: build run logs