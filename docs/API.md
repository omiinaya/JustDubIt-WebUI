# API Reference

## POST /api/dub
Submit video for dubbing.

Request: multipart/form-data with video file and parameters
Response: {"job_id": "abc123", "status": "queued"}

## GET /api/jobs
List all jobs.

## GET /api/job/{id}
Get job status.

## GET /api/download/{id}
Download completed video.

## DELETE /api/delete/{id}
Delete a job.
