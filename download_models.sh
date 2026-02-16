#!/bin/bash
# Model download script for JustDubIt
set -e

MODELS_DIR="/opt/just-dub-it/models"
cd "$MODELS_DIR"

echo "Downloading JustDubIt models..."
echo "This requires HuggingFace authentication. Make sure you have a HF_TOKEN env variable set."

# Install huggingface_hub if not present
pip install -q huggingface-hub

# Download models
if [ ! -f "ltx-2-19b-dev.safetensors" ]; then
    echo "Downloading LTX-2 model..."
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir . --local-dir-use-symlinks False
fi

if [ ! -f "ltx-2-19b-distilled-lora-384.safetensors" ]; then
    echo "Downloading distilled LoRA..."
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-lora-384.safetensors --local-dir . --local-dir-use-symlinks False
fi

if [ ! -f "ltx-2-spatial-upscaler-x2-1.0.safetensors" ]; then
    echo "Downloading spatial upsampler..."
    huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir . --local-dir-use-symlinks False
fi

if [ ! -f "ltx-2-19b-ic-lora-lipdubbing.safetensors" ]; then
    echo "Downloading JustDubit LoRA..."
    huggingface-cli download justdubit/justdubit ltx-2-19b-ic-lora-lipdubbing.safetensors --local-dir . --local-dir-use-symlinks False
fi

echo "Models downloaded successfully!"
