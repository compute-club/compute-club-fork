FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

# Define arguments
ARG USER_NAME
ARG USER_UID
ARG USER_GID

# Configure environment
ENV SHELL=/bin/bash
ENV USER_HOME=/home/$USER_NAME
ENV PROJECT_HOME=$USER_HOME/project

# Create non-root user
RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME

# Install OS packages
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update -y  \
    && apt-get install -yq --no-install-recommends \
        python3.9 \
        python3.9-distutils \
        python3-pip \
        curl \
        git \
        htop

# Set working directory
WORKDIR $PROJECT_HOME
RUN chown -R $USER_UID:$USER_GID $PROJECT_HOME
COPY requirements.txt $PROJECT_HOME

# Set user
USER $USER_NAME

# Install project dependencies
RUN pip3 install -r requirements.txt

# Pull fastchat
RUN git clone https://github.com/lm-sys/FastChat.git
RUN cd FastChat && pip3 install -e .

