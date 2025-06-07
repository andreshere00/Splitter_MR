# Load environment variables from .env if it exists.
ifneq (,$(wildcard .env))
	include .env
	export
endif

.PHONY: help
help:  ## Show this help message.
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build:  ## Build the project.
	@echo "Building the project using uv..."
	@uv build

.PHONY: install
install: ## Install uv CLI and pre-commit.
	@echo "Installing uv CLI and pre-commit..."
	$(UV_INSTALL_CMD)
	@uv add --dev pre-commit
	@uv run pre-commit install

.PHONY: shell
shell:  ## Run a uv shell.
	@echo "Starting a uv shell..."
	uv run shell

.PHONY: clean  ## Clean cache and temporary files
clean:
	@echo "Clean all the cache and temporary files."
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type f -name '*.pyc' -delete
	@find . -type f -name '*.pyo' -delete
	@find . -type f -name '*.pyd' -delete
	@find . -type d -name '*.egg-info' -exec rm -rf {} +
	@find . -type f -name '*.egg' -delete

.PHONY: test ## Run tests and check coverage
test:
	@echo "Run tests and check test coverage."
	@uv run --dev pytest --cov

.PHONY: docs
docs:  ## Deploy Mkdocs server
	@echo "Deploying Mkdocs server..."
	@uv run mkdocs serve

.PHONY: pre-commit
pre-commit:  ## Install pre-commit hooks.
	@echo "Installing pre-commit hooks..."
	uv run pre-commit

.PHONY: format
format:  ## Run pyupgrade, isort, black, and flake8 for code style.
	@echo "Running pyupgrade..."
	uv run --dev pyupgrade --exit-zero
	@echo "Running isort..."
	uv run --dev isort .
	@echo "Running black..."
	uv run --dev black .
	@echo "Running flake8..."
	uv run --dev flake8 --max-line-length=101 --ignore=E203,W291,E501 src/
