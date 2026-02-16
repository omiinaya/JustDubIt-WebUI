from flask import Flask, render_template, request, jsonify
import os
import uuid
from datetime import datetime

app = Flask(__name__)

jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/jobs')
def get_jobs():
    return jsonify({
        'jobs': list(jobs.values()),
        'message': 'Demo mode - Install ML dependencies to enable dubbing'
    })

@app.route('/api/dub', methods=['POST'])
def dub_video():
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        'id': job_id,
        'status': 'demo',
        'message': 'Demo mode. See README.md for full setup.',
        'created_at': datetime.now().isoformat()
    }
    return jsonify({
        'job_id': job_id,
        'status': 'demo',
        'message': 'WebUI is in demo mode. Install models for full functionality.'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'mode': 'demo'})

if __name__ == '__main__':
    print("JustDubIt WebUI - Demo Mode")
    print("Access: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)
