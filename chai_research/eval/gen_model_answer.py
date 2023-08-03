import argparse
import json

import torch
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
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    # TODO: set load_in_8bit=True
    model = GPTJForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
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
