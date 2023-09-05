# LLMs as Judge

## Running model evaluation

1. Set environment variables in the file `.env.template`

Example:
```
MODEL_PATH=EleutherAI/gpt-j-6b
LORA_PATH=
OPENAI_API_KEY=sk-XXXXXXXXXXXX
HUGGINGFACE_TOKEN=hf_XXXXXXXXX
```

Example with LORA path:
```
MODEL_PATH=EleutherAI/gpt-j-6b
LORA_PATH=lwk723/gptj-qlora-peft
OPENAI_API_KEY=sk-XXXXXXXXXXXX
HUGGINGFACE_TOKEN=hf_XXXXXXXXX
```


2. Run docker compose command to run job

```
sudo docker compose up eval
```

You should see results in console:

```
eval-eval-1  | question: 91, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 92, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 93, model: gpt-j-6B, score: 2, judge: gpt-4 
eval-eval-1  | question: 94, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 95, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 96, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 97, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 98, model: gpt-j-6B, score: 3, judge: gpt-4 
eval-eval-1  | question: 99, model: gpt-j-6B, score: 1, judge: gpt-4 
eval-eval-1  | question: 100, model: gpt-j-6B, score: 2, judge: gpt-4 
eval-eval-1  | Average results:
eval-eval-1  |           score
eval-eval-1  | model          
eval-eval-1  | gpt-j-6B    1.4
eval-eval-1  | Done
eval-eval-1 exited with code 0
```

