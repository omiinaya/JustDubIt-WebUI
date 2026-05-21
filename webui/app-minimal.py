import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template

app = Flask(__name__)

jobs = {}


@app.route("/")
def index():  # noqa: ANN201
    return render_template("index.html")


@app.route("/api/jobs")
def get_jobs():  # noqa: ANN201
    return jsonify(
        {
            "jobs": list(jobs.values()),
            "message": "Demo mode - Install ML dependencies to enable dubbing",
        }
    )


@app.route("/api/dub", methods=["POST"])
def dub_video():  # noqa: ANN201
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "status": "demo",
        "message": "Demo mode. See README.md for full setup.",
        "created_at": datetime.now().isoformat(),  # noqa: DTZ005
    }
    return jsonify(
        {
            "job_id": job_id,
            "status": "demo",
            "message": "WebUI is in demo mode. Install models for full functionality.",
        }
    )


@app.route("/health")
def health():  # noqa: ANN201
    return jsonify({"status": "ok", "mode": "demo"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
