FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \\
    git curl ffmpeg libgl1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy files
COPY . .

# Install Python deps (ensure Flask is included)
RUN uv venv --python 3.11 && \\
    . .venv/bin/activate && \\
    uv sync && \\
    pip install flask flask-cors -q

# Create directories
RUN mkdir -p models webui/uploads webui/outputs

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:5000/health 2>/dev/null || exit 1

CMD [".venv/bin/python", "webui/app.py"]
