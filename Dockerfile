# Base image with Python
FROM python:3.12-slim

# Environment variables for Poetry
ENV POETRY_VERSION=2.1.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VENV="/opt/poetry-venv" \
    PATH="/opt/poetry/bin:$PATH"

# Install curl and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglx-mesa0 \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/* 

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry --version

# Create project directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev by default, use --with dev if needed)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy application source
COPY app/ ./app/

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
