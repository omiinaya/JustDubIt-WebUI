#!/bin/bash
# JustDubIt WebUI Installer

echo "========================================"
echo "JustDubIt WebUI Installer"
echo "========================================"
echo ""
echo "Checking requirements..."

# Check disk space
AVAILABLE=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "0")
if [ "$AVAILABLE" -lt 50 ]; then
    echo "WARNING: Low disk space (${AVAILABLE}GB). Recommended: 100GB+"
fi

echo ""
echo "Select installation mode:"
echo "  1) Full (downloads 76GB of AI models)"
echo "  2) Demo (WebUI only, no AI)"
echo "  3) Docker (recommended)"
echo ""
read -p "Choice [1-3]: " CHOICE

if [ "$CHOICE" = "1" ]; then
    echo "Installing Full version..."
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: Set HF_TOKEN first: export HF_TOKEN=your_token"
        exit 1
    fi
    uv sync || pip install -r requirements.txt
    ./download_models.sh
    echo "✓ Full installation complete"
    
elif [ "$CHOICE" = "2" ]; then
    echo "Installing Demo mode..."
    pip install flask flask-cors --quiet 2>/dev/null || apt-get install -y python3-flask
    echo "✓ Demo installed. Run: python webui/app-minimal.py"
    
elif [ "$CHOICE" = "3" ]; then
    echo "Installing via Docker..."
    docker compose up -d
    echo "✓ Docker running. Access: http://localhost:5000"
else
    echo "Invalid choice"
    exit 1
fi

