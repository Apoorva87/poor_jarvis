# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libsndfile1-dev \
    libasound2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the wait script and make it executable
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SOUNDDEVICE_BACKEND=coreaudio

# Run the application with wait script
CMD ["/wait-for-it.sh", "python", "example_parallel_sva.py"] 