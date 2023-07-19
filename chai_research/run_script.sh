#!/bin/bash

# Define project paths
PROJECT_HOME="/home/mrich388/project"
LMGYM_PATH="$PROJECT_HOME/scripts/train/explorations/lmgym/clm_models"

# Activate virtual environment
python3 -m venv env
source env/bin/activate

# Install requirements
pip install -r requirements.txt

# Add directories to Python path
export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME:$LMGYM_PATH"

# Run python script
python3 $LMGYM_PATH/train.py
