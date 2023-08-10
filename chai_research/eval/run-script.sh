#!/bin/bash

set -e

echo "TEST_MODE=$TEST_MODE"

if [ "$TEST_MODE" = "fail" ]; then
  exit 1
fi

echo "Running LLM Judge"

export MODEL_PATH=${MODEL_PATH:-"EleutherAI/gpt-j-6B"}
export MAX_NEW_TOKEN=${MAX_NEW_TOKEN:-"128"}
export OPENAI_API_KEY=$OPENAI_API_KEY
export HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

echo "[ VAR ] MODEL_PATH=$MODEL_PATH"
echo "[ VAR ] MAX_NEW_TOKEN=$MAX_NEW_TOKEN"

python3 gen_model_answer.py \
    --model-path $MODEL_PATH \
    --max-new-token $MAX_NEW_TOKEN

python3 gen_judgment.py \
    --model-path $MODEL_PATH

echo "Done"
