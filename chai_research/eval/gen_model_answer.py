import argparse
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPTJForCausalLM


def load_questions(question_file):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model(model_path, device):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.getenv("HUGGINGFACE_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    # TODO: set load_in_8bit=True
    model = GPTJForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, token=os.getenv("HUGGINGFACE_TOKEN"))
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    args = parser.parse_args()

    # Load questions
    questions = load_questions("question.jsonl")

    # Load model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, tokenizer = load_model(args.model_path, device)

    # Generate answers
    SEP = "\n"
    records = []
    for question in tqdm(questions):
        turns = []
        prompt = ""
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            prompt += qs + SEP
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)

            try:
                output_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.7,
                    max_length=int(args.max_new_token*(j+1))
                )
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            output_ids = output_ids[0][len(input_ids[0]):]
            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            output = output.strip()
            turns.append(output)
            prompt += output + SEP
        
        record = {
            "question_id": question["question_id"],
            "answer_id": question["question_id"],
            "model": args.model_path,
            "choices": [{"index": 0, "turns": turns}],
            "tstamp": time.time(),
        }
        records.append(record)

    # Save answers
    filename = args.model_path.split('/')[-1]
    with open(f"{filename}.jsonl", "a") as fout:
        for record in records:
            fout.write(json.dumps(record) + "\n")
