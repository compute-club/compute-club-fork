FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install OS packages
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update -y  \
    && apt-get install -yq --no-install-recommends \
    gcc \
    python3-dev \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    curl \
    wget \
    git \
    htop

WORKDIR /app
COPY . .
RUN pip3 install -q -r requirements.txt
