import os
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-8B"
save_dir = "/root/autodl-tmp/models/Qwen3-8B"

os.makedirs(save_dir, exist_ok=True)

# 下载到指定目录
local_model_dir = snapshot_download(
    model_id=model_id,
    local_dir=save_dir,
)

print("模型下载完成！")
print("保存路径：", local_model_dir)

# 可选：下载后测试是否能正常加载
tokenizer = AutoTokenizer.from_pretrained(
    local_model_dir,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)

print("模型和 tokenizer 加载成功。")