# =============================================================================
# UAO Neumonía Grupo 1 — automatización de tareas
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help install dev pre-commit-install pre-commit lint format format-check test check run clean

help: ## Muestra esta ayuda
	@echo "Uso: make <objetivo>"
	@echo ""
	@echo "  help               Muestra esta ayuda"
	@echo "  install            Sincroniza el entorno virtual e instala dependencias"
	@echo "  dev                Prepara el entorno de desarrollo (sync + hooks)"
	@echo "  pre-commit-install Instala los hooks de pre-commit"
	@echo "  pre-commit         Ejecuta todos los hooks sobre el repositorio"
	@echo "  lint               Ejecuta Ruff (lint)"
	@echo "  format             Formatea el código con Ruff"
	@echo "  format-check       Verifica el formato sin modificarlo"
	@echo "  test               Ejecuta las pruebas"
	@echo "  check              Verificación completa (lint + formato + pruebas)"
	@echo "  run                Abre la interfaz gráfica"
	@echo "  clean              Limpia cachés y artefactos"

install: ## Sincroniza el entorno virtual e instala dependencias
	uv sync

dev: install pre-commit-install ## Prepara el entorno de desarrollo (sync + hooks)

pre-commit-install: ## Instala los hooks de pre-commit en el repositorio local
	uv run pre-commit install

pre-commit: ## Ejecuta todos los hooks de pre-commit sobre el repositorio
	uv run pre-commit run --all-files

lint: ## Ejecuta Ruff (lint)
	uv run ruff check .

format: ## Formatea el código con Ruff
	uv run ruff format .

format-check: ## Verifica el formato sin modificarlo
	uv run ruff format --check .

test: ## Ejecuta las pruebas
	uv run pytest

check: lint format-check test ## Verificación completa (lint + formato + pruebas)

run: ## Abre la interfaz gráfica
	uv run python detector_neumonia.py

clean: ## Limpia cachés y artefactos
	uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'dist', 'build', 'htmlcov']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
