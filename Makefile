.PHONY: install install-dev test build-go build-cpp clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python package
	pip install .

install-dev: ## Install in development mode
	pip install -e ".[dev]"

test: ## Run all tests
	python3 test_apc.py

build-go: ## Build Go CLI
	go build -o acies-cli .

build-cpp: ## Build C++ library
	cd cpp && make

build: build-go build-cpp ## Build everything

clean: ## Remove build artifacts
	rm -rf acies-cli *.egg-info dist build __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	cd cpp && make clean 2>/dev/null || true

publish: ## Publish to PyPI (requires auth)
	pip install build twine
	python -m build
	twine upload dist/*
