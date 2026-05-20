import os
import subprocess
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "/opt/just-dub-it/webui/uploads"
OUTPUT_FOLDER = "/opt/just-dub-it/webui/outputs"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # noqa: PTH103
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # noqa: PTH103

# Store job status
jobs = {}


def allowed_file(filename):  # noqa: ANN001, ANN201
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():  # noqa: ANN201
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():  # noqa: ANN201
    """Get available models"""
    models_dir = "/opt/just-dub-it/models"
    models = []
    if os.path.exists(models_dir):  # noqa: PTH110
        for f in os.listdir(models_dir):  # noqa: PTH208
            if f.endswith(".safetensors"):
                size = os.path.getsize(os.path.join(models_dir, f)) / (1024**3)  # noqa: PTH118, PTH202
                models.append({"name": f, "size_gb": round(size, 2)})
    return jsonify({"models": models})


@app.route("/api/jobs", methods=["GET"])
def get_jobs():  # noqa: ANN201
    """Get all jobs"""
    return jsonify({"jobs": list(jobs.values())})


@app.route("/api/job/<job_id>", methods=["GET"])
def get_job(job_id):  # noqa: ANN001, ANN201
    """Get job status"""
    if job_id in jobs:
        return jsonify(jobs[job_id])
    return jsonify({"error": "Job not found"}), 404


@app.route("/api/dub", methods=["POST"])
def dub_video():  # noqa: ANN201
    """Submit a dubbing job"""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Supported: mp4, avi, mov, mkv, webm"}), 400

    # Get parameters
    prompt = request.form.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    height = request.form.get("height", "512")
    width = request.form.get("width", "768")
    steps = request.form.get("steps", "30")
    cfg = request.form.get("cfg", "3.0")
    fps = request.form.get("fps", "25")
    seed = request.form.get("seed", "42")

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    filename = f"{job_id}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)  # noqa: PTH118
    file.save(filepath)

    # Create output path
    output_filename = f"{job_id}_dubbed.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)  # noqa: PTH118

    # Store job info
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued for processing",
        "input_file": filename,
        "output_file": output_filename,
        "prompt": prompt,
        "params": {"height": height, "width": width, "steps": steps, "cfg": cfg, "fps": fps, "seed": seed},
        "created_at": datetime.now().isoformat(),  # noqa: DTZ005
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    # Start processing in background
    import threading  # noqa: PLC0415

    thread = threading.Thread(
        target=process_video, args=(job_id, filepath, output_path, prompt, height, width, steps, cfg, fps, seed)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


def process_video(job_id, input_path, output_path, prompt, height, width, steps, cfg, fps, seed) -> None:  # noqa: ANN001
    """Process video in background"""
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = datetime.now().isoformat()  # noqa: DTZ005
    jobs[job_id]["message"] = "Starting video dubbing..."
    jobs[job_id]["progress"] = 10

    try:
        # Set up environment
        env = os.environ.copy()
        env["PATH"] = "/root/.local/bin:" + env.get("PATH", "")
        env["GEMMA_ROOT"] = "/opt/gemma"

        # Build command
        cmd = [
            "bash",
            "-c",
            f'''
cd /opt/just-dub-it &&
source .venv/bin/activate &&
python src/ltx_pipelines/pipeline_justdubit.py \\
    --checkpoint_path /opt/just-dub-it/models/ltx-2-19b-dev.safetensors \\
    --gemma_root /opt/gemma \\
    --distilled_lora_path /opt/just-dub-it/models/ltx-2-19b-distilled-lora-384.safetensors \\
    --distilled_lora_strength 1.0 \\
    --spatial_upsampler_path /opt/just-dub-it/models/ltx-2-spatial-upscaler-x2-1.0.safetensors \\
    --lora /opt/just-dub-it/models/ltx-2-19b-ic-lora-lipdubbing.safetensors \\
    --lora_strength 1.0 \\
    --video_conditioning "{input_path}" 1.0 \\
    --prompt "{prompt}" \\
    --height {height} \\
    --width {width} \\
    --num_inference_steps {steps} \\
    --cfg_guidance_scale {cfg} \\
    --frame_rate {fps} \\
    --seed {seed} \\
    --output_path "{output_path}" 2>&1
            ''',
        ]

        jobs[job_id]["message"] = "Running AI dubbing (this may take 10-30 minutes)..."
        jobs[job_id]["progress"] = 20

        # Run process
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env
        )

        # Monitor progress
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            # Update progress based on output
            if "step" in line.lower() and "/" in line:
                try:
                    # Try to parse step progress
                    parts = line.split("/")
                    if len(parts) >= 2:
                        current = int(parts[0].split()[-1])
                        total = int(parts[1].split()[0])
                        progress = 20 + int((current / total) * 70)
                        jobs[job_id]["progress"] = min(progress, 90)
                        jobs[job_id]["message"] = f"Processing step {current}/{total}..."
                except Exception:
                    pass

        process.wait()

        if process.returncode == 0 and os.path.exists(output_path):  # noqa: PTH110
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Video dubbing completed!"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()  # noqa: DTZ005
        else:
            error_msg = "".join(output_lines[-20:]) if output_lines else "Unknown error"
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Processing failed: {error_msg}"
            jobs[job_id]["error"] = error_msg

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {e!s}"
        jobs[job_id]["error"] = str(e)


@app.route("/api/download/<job_id>")
def download_video(job_id):  # noqa: ANN001, ANN201
    """Download completed video"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed"}), 400

    output_path = os.path.join(OUTPUT_FOLDER, job["output_file"])  # noqa: PTH118
    if not os.path.exists(output_path):  # noqa: PTH110
        return jsonify({"error": "Output file not found"}), 404

    return send_file(output_path, as_attachment=True, download_name=f"dubbed_{job['input_file']}")


@app.route("/api/delete/<job_id>", methods=["DELETE"])
def delete_job(job_id):  # noqa: ANN001, ANN201
    """Delete a job and its files"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]

    # Delete files
    input_path = os.path.join(UPLOAD_FOLDER, job["input_file"])  # noqa: PTH118
    output_path = os.path.join(OUTPUT_FOLDER, job["output_file"])  # noqa: PTH118

    if os.path.exists(input_path):  # noqa: PTH110
        os.remove(input_path)  # noqa: PTH107
    if os.path.exists(output_path):  # noqa: PTH110
        os.remove(output_path)  # noqa: PTH107

    # Remove from jobs dict
    del jobs[job_id]

    return jsonify({"message": "Job deleted"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
