import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_inference(model_dir: str, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 优先使用 GPU；当前环境若驱动不匹配会自动回退到 CPU
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
    )
    device = "cuda" if use_cuda else "cpu"
    model = model.to(device)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference with Qwen3-4B.")
    parser.add_argument(
        "--model-dir",
        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507",
        help="Local model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句中文介绍一下你自己。",
        help="User prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    print("model_dir:", args.model_dir)
    print("cuda available:", torch.cuda.is_available())
    answer = run_inference(args.model_dir, args.prompt, args.max_new_tokens)
    print("\n--- OUTPUT ---")
    print(answer)


if __name__ == "__main__":
    main()
