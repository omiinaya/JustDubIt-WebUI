# Installation Guide

## System Requirements

- GPU: NVIDIA GPU with CUDA 12.0+ (RTX 3090 or better)
- RAM: 16GB minimum
- Storage: 100GB free space
- OS: Ubuntu 22.04 or Debian 12

## Installation

### Proxmox Container

```bash
pct create 301 local:vztmpl/debian-12-standard_12.7-1_amd64.tar.zst \\
  --hostname just-dub-it --cores 8 --memory 16384 \\
  --rootfs local-zfs:128 --net0 bridge=vmbr0,ip=dhcp --start 1

pct enter 301
apt-get update && apt-get install -y git curl ffmpeg python3-dev

cd /opt
git clone https://github.com/omiinaya/JustDubIt-WebUI.git
cd JustDubIt-WebUI

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11 && source .venv/bin/activate
uv sync

export HF_TOKEN=your_token
./download_models.sh

systemctl enable --now just-dubit-webui
```

Access at http://your-server:5000

## Model Access Requirements

### Gemma Text Encoder License

Before downloading models, you must accept Google's license terms for the Gemma text encoder:

1. Visit: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Click **"Accept License"** or **"Request Access"**
3. Wait for approval (usually instant)
4. Verify access: `huggingface-cli whoami`

**Without this step, downloads will fail with 403 Forbidden error.**
