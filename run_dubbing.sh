#!/bin/bash
# JustDubIt video dubbing script
set -e

export PATH="$HOME/.local/bin:$PATH"
cd /opt/just-dub-it

# Check if models exist
MODELS_DIR="/opt/just-dub-it/models"
if [ ! -f "$MODELS_DIR/ltx-2-19b-dev.safetensors" ]; then
    echo "Error: Models not found. Run ./download_models.sh first"
    exit 1
fi

# Default values
CHECKPOINT="$MODELS_DIR/ltx-2-19b-dev.safetensors"
GEMMA_ROOT="${GEMMA_ROOT:-/opt/gemma}"  # User needs to set this
SPATIAL_UPSAMPLER="$MODELS_DIR/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA="$MODELS_DIR/ltx-2-19b-distilled-lora-384.safetensors"
JUSTDUBIT_LORA="$MODELS_DIR/ltx-2-19b-ic-lora-lipdubbing.safetensors"

# Parse arguments
VIDEO_INPUT=""
PROMPT=""
OUTPUT="./output.mp4"
HEIGHT=512
WIDTH=768
STEPS=30
CFG=3.0
FPS=25

while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_INPUT="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --cfg)
            CFG="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VIDEO_INPUT" ] || [ -z "$PROMPT" ]; then
    echo "Usage: ./run_dubbing.sh --video <input.mp4> --prompt '<prompt>' [options]"
    echo ""
    echo "Options:"
    echo "  --video     Input video file path (required)"
    echo "  --prompt    Text prompt describing desired output (required)"
    echo "  --output    Output file path (default: ./output.mp4)"
    echo "  --height    Stage 1 height (default: 512, final output: 1024)"
    echo "  --width     Stage 1 width (default: 768, final output: 1536)"
    echo "  --steps     Number of inference steps (default: 30)"
    echo "  --cfg       CFG guidance scale (default: 3.0)"
    echo "  --fps       Output frame rate (default: 25)"
    exit 1
fi

echo "Running JustDubIt video dubbing..."
echo "Input: $VIDEO_INPUT"
echo "Output: $OUTPUT"
echo "Prompt: $PROMPT"

uv run python src/ltx_pipelines/pipeline_justdubit.py \
    --checkpoint_path "$CHECKPOINT" \
    --gemma_root "$GEMMA_ROOT" \
    --distilled_lora_path "$DISTILLED_LORA" \
    --distilled_lora_strength 1.0 \
    --spatial_upsampler_path "$SPATIAL_UPSAMPLER" \
    --lora "$JUSTDUBIT_LORA" \
    --lora_strength 1.0 \
    --video_conditioning "$VIDEO_INPUT" 1.0 \
    --prompt "$PROMPT" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $STEPS \
    --cfg_guidance_scale $CFG \
    --frame_rate $FPS \
    --seed 42 \
    --output_path "$OUTPUT"

echo "Dubbing complete! Output saved to: $OUTPUT"
