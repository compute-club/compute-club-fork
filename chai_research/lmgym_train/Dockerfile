FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Define arguments
ARG USER_NAME
ARG USER_UID
ARG USER_GID

# Configure environment
ENV SHELL=/bin/bash
ENV USER_HOME=/home/$USER_NAME
ENV PROJECT_HOME=$USER_HOME/project
ENV PATH="$USER_HOME/.local/bin:$PATH"

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
        wget \
        git \
        htop

# Set working directory
WORKDIR $PROJECT_HOME

# copy files with root user
COPY ./lmgym_train/requirements.txt $PROJECT_HOME
COPY ./lmgym_train/train $PROJECT_HOME/train

# Make script executable while still root user
RUN chmod +x $PROJECT_HOME/train/explorations/lmgym_files/hh_demo.sh
RUN chmod +x $PROJECT_HOME/train/explorations/lmgym_files/train.py

# change ownership of the project directory
RUN chown -R $USER_UID:$USER_GID $PROJECT_HOME

# Set user
USER $USER_NAME

# Install project dependencies
RUN pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -q -r requirements.txt

# Clone lmgym repository and set PYTHONPATH
RUN git clone https://github.com/chai-research/lmgym.git
ENV PYTHONPATH $PROJECT_HOME/lmgym

