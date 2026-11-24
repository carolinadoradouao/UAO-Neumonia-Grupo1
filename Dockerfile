FROM python:3.13-slim

# uv como único gestor de dependencias y entornos (sin pip).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Librerías del sistema necesarias para OpenCV, TensorFlow y Tkinter.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tk \
        xvfb && \
    rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Instalar dependencias desde el lockfile (capa cacheable y reproducible).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev

# 2) Copiar el código de la aplicación.
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

# El modelo entrenado no se versiona: móntalo en /app/model al ejecutar el contenedor.
CMD ["python", "detector_neumonia.py"]
