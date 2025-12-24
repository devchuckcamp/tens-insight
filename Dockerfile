# Dockerfile for Tens-Insight ML Pipeline
# Use official TensorFlow Docker image 2.17.0 (stable, CPU-only)
FROM tensorflow/tensorflow:2.17.0

WORKDIR /app

# Install system dependencies (postgresql-client for debugging)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install additional Python dependencies
# Note: TensorFlow is already included in the base image
COPY requirements.txt .
RUN pip install --no-cache-dir \
    'numpy>=1.24.0,<2.0.0' \
    'pandas>=2.0.0,<3.0.0' \
    'scikit-learn>=1.3.0,<2.0.0' \
    'sqlalchemy>=2.0.0,<3.0.0' \
    'psycopg2-binary>=2.9.0,<3.0.0' \
    'joblib>=1.3.0,<2.0.0' \
    'python-dotenv>=1.0.0,<2.0.0' \
    'apscheduler>=3.10.0,<4.0.0' \
    'pytz>=2023.3'

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "-m", "src.training.train_churn"]
