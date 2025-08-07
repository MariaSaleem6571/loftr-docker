FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Build args for host user/group ID
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create group and user with specified IDs
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

# Set working directory inside user's home
WORKDIR /home/appuser/app

# Copy files
COPY requirements.txt .
COPY match.py .
COPY images/ ./images/

# Install Python dependencies
RUN pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt

# Switch to the created user
USER appuser

# Default command
CMD ["python", "match.py"]

