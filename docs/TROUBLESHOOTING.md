# Troubleshooting

## CUDA Out of Memory
Lower resolution: --height 384 --width 512
Reduce steps: --num_inference_steps 20

## Model Download Fails
Check HF_TOKEN is set and valid
Request access at huggingface.co

## Service Won't Start
systemctl status just-dubit-webui
journalctl -u just-dubit-webui -f

## Long Processing
Check GPU: nvidia-smi
Expected: 10-30 min per video
