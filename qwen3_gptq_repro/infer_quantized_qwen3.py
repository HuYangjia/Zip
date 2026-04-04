import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_quantized_model(model_dir: str, quantized_weights: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    state_dict = torch.load(quantized_weights, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[warn] missing_keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[warn] unexpected_keys: {len(unexpected_keys)}")
    return model


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


def main():
    parser = argparse.ArgumentParser(description="Load GPTQ-quantized 4-bit Qwen3 state_dict and run inference.")
    parser.add_argument(
        "--model-dir",
        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507",
        help="Base model directory.",
    )
    parser.add_argument(
        "--quantized-weights",
        default="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output/qwen3-4b-instruct-2507-gptq-4bit.pt",
        help="Path to quantized state_dict (.pt).",
    )
    parser.add_argument("--prompt", default="请用一句话介绍你自己。", help="User prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_quantized_model(args.model_dir, args.quantized_weights, dtype=dtype)

    print("cuda available:", torch.cuda.is_available())
    print("quantized weights:", args.quantized_weights)
    answer = run_inference(model, tokenizer, args.prompt, args.max_new_tokens)
    print("\n--- OUTPUT ---")
    print(answer)


if __name__ == "__main__":
    main()
