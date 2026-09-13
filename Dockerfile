# Serves the fine-tuned tool-planner model over HTTP (serve/app.py).
#
# The base model (Qwen2.5-1.5B-Instruct, ~6GB) is NOT baked into this image --
# it's downloaded from Hugging Face Hub at container startup and cached in
# /root/.cache/huggingface. For repeated runs (local Docker or a redeployed
# EC2 instance), mount a volume there to avoid re-downloading it every time:
#   docker run -v hf-cache:/root/.cache/huggingface ...
#
# Only the LoRA adapter (~90MB) ships in the image -- the training
# checkpoints (checkpoint-*/, optimizer state, ~212MB each) are excluded via
# .dockerignore since serving only needs the final adapter weights.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# torch first, from PyPI's CPU-only wheel index: the default PyPI wheel bundles
# several GB of NVIDIA CUDA libraries (cudnn, cublas, nccl, triton, ...) that
# this CPU-only serving container will never use, ballooning the image from
# ~1.5GB to ~9.5GB for nothing. Installing the CPU build here first means the
# later `pip install -r requirements.txt` sees the pin already satisfied and
# leaves it alone rather than pulling the GPU build back in.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/ engine/
COPY train/prompt.py train/prompt.py
COPY serve/ serve/
COPY checkpoints/lora-adapter-v2/adapter_config.json checkpoints/lora-adapter-v2/adapter_config.json
COPY checkpoints/lora-adapter-v2/adapter_model.safetensors checkpoints/lora-adapter-v2/adapter_model.safetensors
COPY checkpoints/lora-adapter-v2/tokenizer.json checkpoints/lora-adapter-v2/tokenizer.json
COPY checkpoints/lora-adapter-v2/tokenizer_config.json checkpoints/lora-adapter-v2/tokenizer_config.json
COPY checkpoints/lora-adapter-v2/chat_template.jinja checkpoints/lora-adapter-v2/chat_template.jinja

ENV ADAPTER_PATH=/app/checkpoints/lora-adapter-v2
ENV PYTHONUNBUFFERED=1

EXPOSE 8888
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8888"]
