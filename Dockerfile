FROM python:3.11-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  build-essential \
  python3-dev \
  libopenblas-dev \
  liblapack-dev \
  git \
  && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the application port
EXPOSE 8082

# Set default command
CMD ["python", "app.py"]