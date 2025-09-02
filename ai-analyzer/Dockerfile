# Use slim Python base image
FROM python:3.9-slim

# Install security updates + dependencies for gunicorn/psycopg2/etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl tini && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependency file first (for layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5001

# Expose port
EXPOSE 5001

ENTRYPOINT ["/usr/bin/tini", "--", "gunicorn", "--bind=0.0.0.0:5001", "app:app"]
