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

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

WORKDIR /home/appuser/app

COPY requirements.txt .
COPY match.py .
COPY images/ ./images/

RUN pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt

USER appuser

#CMD ["python", "match.py"]
CMD ["python", "homography_batch.py"]


