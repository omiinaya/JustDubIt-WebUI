# JustDubIt WebUI

A complete AI-powered video dubbing solution with an intuitive web interface. Translate videos with synchronized lip movements and natural speech using state-of-the-art AI models.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-AGPL--3.0-orange)](LICENSE)

---

## 🎯 Features

- **🎬 Video Upload** - Drag & drop interface for video files
- **🌍 Translation** - Translate speech to any language with natural lip sync
- **⚙️ Quality Settings** - Adjustable resolution, inference steps, and guidance
- **📊 Progress Tracking** - Real-time job status and progress bars
- **💾 Download** - Easy download of completed dubbed videos
- **🎨 Modern UI** - Dark theme with responsive design
- **🔒 Private** - Self-hosted, no data leaves your server

---

## 🚀 Quick Start

### Prerequisites

- **GPU:** NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- **RAM:** 16GB+ system memory
- **Storage:** 100GB+ free space for models
- **OS:** Linux (Ubuntu/Debian preferred)

### Installation

#### Option 1: Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/omiinaya/JustDubIt-WebUI.git
cd JustDubIt-WebUI

# Build and run
docker-compose up -d

# Access at http://localhost:5000
```

#### Option 2: Manual Installation

```bash
# Clone repository
git clone https://github.com/omiinaya/JustDubIt-WebUI.git
cd JustDubIt-WebUI

# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv sync

# Download models (requires HuggingFace token)
export HF_TOKEN=your_huggingface_token
./download_models.sh

# Start WebUI
python webui/app.py
```

---

## 📋 Model Downloads

The following models are required and will be downloaded automatically:

| Model | Size | Purpose |
|-------|------|---------|
| LTX-2 Base | 40GB | Core video generation model |
| LTX-2 Distilled LoRA | 7GB | Stage 2 refinement |
| Spatial Upsampler | 1GB | 2x upscaling |
| JustDubIt LoRA | 5GB | Lip dubbing adaptation |
| Gemma Text Encoder | 23GB | Text understanding |

**Total:** ~76GB

---

## 💻 Usage

### Web Interface

1. Open http://your-server:5000 in your browser
2. Upload a video file (MP4, AVI, MOV, MKV, WEBM)
3. Enter translation prompt in format:
   ```
   The person is speaking Spanish, saying: "Hola mundo"
   ```
4. Adjust quality settings (optional):
   - **Height/Width:** Stage 1 resolution (default: 512x768, final: 1024x1536)
   - **Steps:** More steps = higher quality (default: 30)
   - **CFG Scale:** Higher = more adherence to prompt (default: 3.0)
5. Click "Start Dubbing"
6. Wait 10-30 minutes for processing
7. Download your dubbed video

### Command Line

```bash
# Using the wrapper script
./run_dubbing.sh \\
    --video input.mp4 \\
    --prompt "The person is speaking French, saying: 'Bonjour le monde'" \\
    --output output.mp4 \\
    --height 512 \\
    --width 768 \\
    --steps 30
```

---

## 🏗️ Architecture

```
JustDubIt-WebUI/
├── webui/
│   ├── app.py              # Flask backend
│   ├── templates/
│   │   └── index.html      # Web interface
│   ├── uploads/            # Temporary upload storage
│   └── outputs/            # Generated videos
├── packages/
│   ├── ltx-core/           # Core LTX-2 model
│   ├── ltx-pipelines/      # Video processing pipelines
│   └── ltx-trainer/        # Training tools
├── models/                 # Downloaded model files
├── download_models.sh      # Model download script
├── run_dubbing.sh          # CLI wrapper
├── pyproject.toml           # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | Required |
| `GEMMA_ROOT` | Path to Gemma text encoder | `/opt/gemma` |
| `FLASK_PORT` | WebUI port | `5000` |
| `FLASK_HOST` | WebUI host | `0.0.0.0` |

### Nginx Reverse Proxy

To expose the WebUI via HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name dub.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }
}
```

---

## 📚 Documentation

- [Installation Guide](docs/INSTALL.md) - Detailed installation instructions
- [Usage Guide](docs/USAGE.md) - How to use the WebUI and CLI
- [API Reference](docs/API.md) - REST API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

---

## 🛠️ Development

### Local Development

```bash
# Clone repo
git clone https://github.com/omiinaya/JustDubIt-WebUI.git
cd JustDubIt-WebUI

# Install dev dependencies
uv sync --dev

# Run in development mode
export FLASK_ENV=development
python webui/app.py
```

### Testing

```bash
# Run tests
pytest tests/
```

### Building

```bash
# Build Docker image
docker build -t justdubit-webui .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models justdubit-webui
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

The underlying JustDubIt model is also AGPL-3.0 licensed.

---

## 🙏 Acknowledgments

- [JustDubIt](https://github.com/justdubit/just-dub-it) - Original video dubbing model
- [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) - Base video generation model
- [Gemma](https://huggingface.co/google/gemma-3-12b-it) - Text encoder
- [Flask](https://flask.palletsprojects.com) - Web framework

---

## 📧 Contact

- **GitHub Issues:** https://github.com/omiinaya/JustDubIt-WebUI/issues
- **Email:** omar@mrxlab.net

---

<p align="center">
  Made with ❤️ by Omar Minaya
</p>
