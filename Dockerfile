# Multi-stage build for Medical RAG Assistant with uv
FROM python:3.11-slim as builder

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project metadata
COPY pyproject.toml .

# Sync dependencies with uv
RUN uv sync --frozen


# ============================================================================
# Runtime Stage
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/tmp/torch \
    HF_HOME=/tmp/huggingface

# Copy application code
COPY app ./app
COPY data ./data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command: Run Chainlit
CMD ["chainlit", "run", "app/api/chainlit_app.py", "--host", "0.0.0.0", "--port", "8001"]

EXPOSE 8001
