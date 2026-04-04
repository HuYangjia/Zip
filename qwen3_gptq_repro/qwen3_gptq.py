import argparse
import importlib.util
import inspect
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


REPO_ROOT = Path(__file__).resolve().parents[1]
GPTQ_DIR = REPO_ROOT / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

from gptq import GPTQ  # noqa: E402
from modelutils import find_layers  # noqa: E402
from quant import Quantizer  # noqa: E402


def load_custom_model_class(modeling_file: str):
    file_path = Path(modeling_file).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"modeling file not found: {file_path}")

    module_name = "transformers.models.qwen3.modeling_qwen3_custom_update"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model_cls = getattr(module, "Qwen3ForCausalLM")
    return model_cls


def register_custom_model(custom_model_cls) -> None:
    try:
        AutoModelForCausalLM.register(Qwen3Config, custom_model_cls, exist_ok=True)
    except TypeError:
        AutoModelForCausalLM.register(Qwen3Config, custom_model_cls)


def build_temp_calib_text(seed: int, text_path: Path, min_lines: int = 3000) -> str:
    random.seed(seed)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    subjects = ["AI", "quantization", "Qwen3", "matrix", "token", "inference", "language model"]
    verbs = ["improves", "compresses", "predicts", "summarizes", "transforms", "analyzes", "generates"]
    objs = ["latency", "memory", "accuracy", "throughput", "weights", "activations", "prompts"]
    lines = []
    for i in range(min_lines):
        s = random.choice(subjects)
        v = random.choice(verbs)
        o = random.choice(objs)
        lines.append(f"[tmp-calib-{i:04d}] {s} {v} {o}.")
    corpus = "\n".join(lines)
    text_path.write_text(corpus, encoding="utf-8")
    return corpus


def sample_trainloader_from_tokens(input_ids: torch.Tensor, nsamples: int, seqlen: int, seed: int):
    if input_ids.shape[1] <= seqlen:
        raise ValueError(
            f"token count ({input_ids.shape[1]}) is not enough for seqlen={seqlen}; "
            "increase source text or reduce --seqlen."
        )
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_wikitext2_or_fallback_loader(
    tokenizer,
    model_dir: str,
    nsamples: int,
    seqlen: int,
    seed: int,
    output_dir: Path,
    local_wikitext2_dir: str = "",
):
    source_info = {
        "calib_source": "wikitext2",
        "fallback_used": False,
        "reason": "",
        "temp_text_file": "",
        "local_wikitext2_dir": "",
    }

    def _load_local_wikitext2_text(local_dir: Path) -> str:
        from datasets import load_dataset, load_from_disk

        local_dir = local_dir.resolve()
        train_arrow = local_dir / "wikitext-train.arrow"
        if train_arrow.exists():
            ds = load_dataset("arrow", data_files={"train": str(train_arrow)}, split="train")
            text_col = "text" if "text" in ds.column_names else ds.column_names[0]
            return "\n\n".join(ds[text_col])

        loaded = load_from_disk(str(local_dir))
        if hasattr(loaded, "keys") and "train" in loaded:
            train_ds = loaded["train"]
        else:
            train_ds = loaded
        text_col = "text" if "text" in train_ds.column_names else train_ds.column_names[0]
        return "\n\n".join(train_ds[text_col])

    if local_wikitext2_dir:
        local_dir = Path(local_wikitext2_dir)
        try:
            joined_text = _load_local_wikitext2_text(local_dir)
            trainenc = tokenizer(joined_text, return_tensors="pt")
            trainloader = sample_trainloader_from_tokens(trainenc.input_ids, nsamples, seqlen, seed)
            source_info["calib_source"] = "local_wikitext2_dir"
            source_info["local_wikitext2_dir"] = str(local_dir.resolve())
            return trainloader, source_info
        except Exception as exc:
            source_info["fallback_used"] = True
            source_info["reason"] = f"local_wikitext2_dir_failed: {exc}"
            source_info["local_wikitext2_dir"] = str(local_dir.resolve())

    try:
        from datasets import load_dataset

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined_text = "\n\n".join(traindata["text"])
        trainenc = tokenizer(joined_text, return_tensors="pt")
        trainloader = sample_trainloader_from_tokens(trainenc.input_ids, nsamples, seqlen, seed)
        return trainloader, source_info
    except Exception as exc:  # fallback path required by user
        source_info["fallback_used"] = True
        source_info["reason"] = str(exc)
        source_info["calib_source"] = "generated_temp_text"
        temp_text_file = output_dir / "tmp_generated_calib.txt"
        source_info["temp_text_file"] = str(temp_text_file)
        text = build_temp_calib_text(seed=seed, text_path=temp_text_file)
        trainenc = tokenizer(text, return_tensors="pt")
        trainloader = sample_trainloader_from_tokens(trainenc.input_ids, nsamples, seqlen, seed)
        return trainloader, source_info


def get_qwen3(model_dir: str, dtype: torch.dtype):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    max_seq = getattr(model.config, "max_position_embeddings", 2048)
    model.seqlen = min(2048, int(max_seq))
    return model


@torch.no_grad()
def qwen3_sequential(
    model,
    dataloader,
    dev: torch.device,
    nsamples: int,
    percdamp: float,
    groupsize: int,
    sym: bool,
    act_order: bool,
    true_sequential: bool,
):
    print("Starting sequential GPTQ (Qwen3) ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "cache_position": None,
        "position_embeddings": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_type = getattr(module, "attention_type", "full_attention")

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["cache_position"] = kwargs.get("cache_position")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    cache_position = cache["cache_position"]
    position_embeddings = cache["position_embeddings"]

    print("Calibration capture ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(4, perchannel=True, sym=sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(nsamples):
                layer_out = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                outs[j] = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Layer {i} -> {name}")
                gptq[name].fasterquant(
                    percdamp=percdamp,
                    groupsize=groupsize,
                    actorder=act_order,
                    static_groups=False,
                )
                quantizers[f"model.layers.{i}.{name}"] = {
                    "bits": 4,
                    "groupsize": groupsize,
                    "sym": sym,
                    "act_order": act_order,
                }
                gptq[name].free()

        for j in range(nsamples):
            layer_out = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            outs[j] = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 GPTQ 4-bit quantization (WikiText first, fallback local text).")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/zhou/Documents/yangjia/zip/models/Qwen3-4B-Instruct-2507",
        help="Local model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/zhou/Documents/yangjia/zip/qwen3_gptq_repro/output",
        help="Directory to save quantized artifacts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--sym", action="store_true", help="Enable symmetric quantization.")
    parser.add_argument("--act-order", action="store_true", help="Enable activation-order GPTQ heuristic.")
    parser.add_argument("--true-sequential", action="store_true", help="Enable true sequential group quantization.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Load dtype for model weights.",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=4,
        help="Fixed to 4 in this reproduction. Any non-4 value will raise error.",
    )
    parser.add_argument(
        "--custom-modeling-file",
        type=str,
        default="",
        help="Optional custom modeling file path, e.g. modeling_qwn3_update.py.",
    )
    parser.add_argument(
        "--local-wikitext2-dir",
        type=str,
        default="",
        help="Optional local WikiText2 directory (contains wikitext-train.arrow or load_from_disk output).",
    )
    parser.add_argument(
        "--init-state-dict",
        type=str,
        default="",
        help="Optional state_dict path. If provided, load it before GPTQ (e.g. smoothed weights).",
    )
    parser.add_argument(
        "--smooth-scales-path",
        type=str,
        default="",
        help="Optional SmoothQuant scales artifact path for metadata tracing.",
    )
    parser.add_argument(
        "--smooth-metadata-path",
        type=str,
        default="",
        help="Optional SmoothQuant metadata artifact path for metadata tracing.",
    )
    return parser.parse_args()


def dtype_from_str(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    if args.wbits != 4:
        raise ValueError("This reproduction supports 4-bit only. Please set --wbits 4.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.custom_modeling_file:
        custom_model_cls = load_custom_model_class(args.custom_modeling_file)
        register_custom_model(custom_model_cls)
        print("registered custom class:", custom_model_cls)
        print("registered from file  :", inspect.getfile(custom_model_cls))

    dtype = dtype_from_str(args.dtype)
    model = get_qwen3(args.model_dir, dtype=dtype)
    model.eval()
    init_state_dict_path = ""
    if args.init_state_dict:
        init_state_dict_path = str(Path(args.init_state_dict).resolve())
        loaded = torch.load(init_state_dict_path, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            loaded = loaded["state_dict"]
        try:
            model.load_state_dict(loaded, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"failed to load --init-state-dict {init_state_dict_path}: {exc}") from exc
        print("initialized model from state_dict:", init_state_dict_path)

    if args.seqlen > model.seqlen:
        print(f"[warn] --seqlen {args.seqlen} > model.seqlen {model.seqlen}; use model.seqlen.")
        args.seqlen = model.seqlen
    model.seqlen = args.seqlen

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    trainloader, source_info = get_wikitext2_or_fallback_loader(
        tokenizer=tokenizer,
        model_dir=args.model_dir,
        nsamples=args.nsamples,
        seqlen=model.seqlen,
        seed=args.seed,
        output_dir=output_dir,
        local_wikitext2_dir=args.local_wikitext2_dir,
    )
    print("calibration source:", source_info["calib_source"])
    if source_info["fallback_used"]:
        print("[fallback] local/wiki load failed, temporary generated text is used.")
        print("[fallback] reason:", source_info["reason"])
        print("[fallback] text file:", source_info["temp_text_file"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPTQ script.")
    dev = torch.device("cuda:0")

    tick = time.time()
    quantizers = qwen3_sequential(
        model=model,
        dataloader=trainloader,
        dev=dev,
        nsamples=args.nsamples,
        percdamp=args.percdamp,
        groupsize=args.groupsize,
        sym=args.sym,
        act_order=args.act_order,
        true_sequential=args.true_sequential,
    )
    total_sec = time.time() - tick
    print(f"quantization time: {total_sec:.2f}s")

    weights_path = output_dir / "qwen3-4b-instruct-2507-gptq-4bit.pt"
    torch.save(model.state_dict(), weights_path)

    metadata = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "wbits": 4,
        "nsamples": args.nsamples,
        "seqlen": model.seqlen,
        "percdamp": args.percdamp,
        "groupsize": args.groupsize,
        "sym": args.sym,
        "act_order": args.act_order,
        "true_sequential": args.true_sequential,
        "dtype": args.dtype,
        "init_state_dict": {
            "used": bool(args.init_state_dict),
            "path": init_state_dict_path,
        },
        "smooth_artifacts": {
            "smooth_scales_path": str(Path(args.smooth_scales_path).resolve()) if args.smooth_scales_path else "",
            "smooth_metadata_path": str(Path(args.smooth_metadata_path).resolve()) if args.smooth_metadata_path else "",
        },
        "calibration": source_info,
        "num_quantized_linear_layers": len(quantizers),
        "elapsed_seconds": total_sec,
        "weights_path": str(weights_path),
    }
    meta_path = output_dir / "qwen3_gptq_4bit_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved weights:", weights_path)
    print("saved meta   :", meta_path)


if __name__ == "__main__":
    main()
