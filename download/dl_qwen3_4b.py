import os
from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-4B-Instruct-2507"
local_dir = "models/Qwen3-4B-Instruct-2507"
endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# 可选：自动读取系统代理（如 Clash/V2Ray），未设置则不走本地代理
http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
proxies = {}
if http_proxy:
    proxies["http"] = http_proxy
if https_proxy:
    proxies["https"] = https_proxy
if not proxies:
    proxies = None

local_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    endpoint=endpoint,
    proxies=proxies,
)

print("使用端点：", endpoint)
if proxies:
    print("使用代理：", proxies)
print("下载完成，路径：", local_path)