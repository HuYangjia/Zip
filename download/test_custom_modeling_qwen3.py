import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def load_custom_model_class(modeling_file: str):
    file_path = Path(modeling_file).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"modeling file not found: {file_path}")

    # 关键点：使用 transformers 包命名空间加载，保证 `...` 相对导入可解析
    module_name = "transformers.models.qwen3.modeling_qwen3_custom"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    # 提前挂到 sys.modules，避免 transformers 的反射/文档装饰器把它当成 built-in class
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model_cls = getattr(module, "Qwen3ForCausalLM")
    return model_cls


def register_custom_model(custom_model_cls) -> None:
    # 将 qwen3 配置映射到你的自定义实现，保持 AutoModel 调用方式不变
    try:
        AutoModelForCausalLM.register(Qwen3Config, custom_model_cls, exist_ok=True)
    except TypeError:
        AutoModelForCausalLM.register(Qwen3Config, custom_model_cls)


def run_inference(model_dir: str, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
    ).to(device)
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
    parser = argparse.ArgumentParser(description="Test inference with custom exported Qwen3 model implementation.")
    parser.add_argument(
        "--model-dir",
        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507",
        help="Local model directory.",
    )
    parser.add_argument(
        "--modeling-file",
        default="/home/zhou/Documents/yangjia/zip/modeling_qwen3.py",
        help="Path to custom modeling_qwen3.py file.",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句话说明你是否成功加载了自定义模型实现。",
        help="User prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    custom_model_cls = load_custom_model_class(args.modeling_file)
    register_custom_model(custom_model_cls)

    print("registered model class:", custom_model_cls)
    print("registered model file :", inspect.getfile(custom_model_cls))
    print("cuda available        :", torch.cuda.is_available())

    answer = run_inference(args.model_dir, args.prompt, args.max_new_tokens)
    print("\n--- OUTPUT ---")
    print(answer)


if __name__ == "__main__":
    main()
