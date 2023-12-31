#!/bin/bash

set -e

echo "TEST_MODE=$TEST_MODE"

if [ "$TEST_MODE" = "fail" ]; then
  exit 1
fi

echo "Running LLM Judge"

export MODEL_PATH=${MODEL_PATH:-"EleutherAI/gpt-j-6b"}
export LORA_PATH=${LORA_PATH:-""}
export MAX_NEW_TOKEN=${MAX_NEW_TOKEN:-"1024"}
export OPENAI_API_KEY=$OPENAI_API_KEY
export HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

echo "[ VAR ] MODEL_PATH=$MODEL_PATH"
echo "[ VAR ] LORA_PATH=$LORA_PATH"
echo "[ VAR ] MAX_NEW_TOKEN=$MAX_NEW_TOKEN"

if [ -z "$MODEL_PATH" ]; then
  echo "MODEL_PATH is not set"
  exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set"
  exit 1
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "HUGGINGFACE_TOKEN is not set"
  exit 1
fi

if [ -z "$LORA_PATH" ]; then
  echo "LORA_PATH is not set"
  python3 gen_model_answer.py \
      --model-path $MODEL_PATH \
      --max-new-token $MAX_NEW_TOKEN

  python3 gen_judgment.py \
      --model-path $MODEL_PATH
else
  echo "LORA_PATH is set"
  python3 gen_model_answer.py \
      --model-path $MODEL_PATH \
      --lora-path $LORA_PATH \
      --max-new-token $MAX_NEW_TOKEN

  python3 gen_judgment.py \
      --model-path $MODEL_PATH \
      --lora-path $LORA_PATH
fi

echo "Done"
