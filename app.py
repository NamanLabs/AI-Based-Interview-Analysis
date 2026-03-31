import os
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from loguru import logger

pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        from pipeline import InterviewAnalysisPipeline
        pipeline_instance = InterviewAnalysisPipeline()
    return pipeline_instance


app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   
UPLOAD_FOLDER  = Path("data/uploads")
RESULTS_FOLDER = Path("data/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "Interview Analyzer API"})


@app.route("/analyze", methods=["POST"])
def analyze_video():
    """
    Upload a video file for analysis.
    Returns full JSON report including all 4 scores.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # Save upload
    job_id    = str(uuid.uuid4())[:8]
    filename  = secure_filename(f"{job_id}_{file.filename}")
    save_path = UPLOAD_FOLDER / filename
    file.save(str(save_path))
    logger.info(f"Video saved: {save_path} (job: {job_id})")

    try:
        pipe   = get_pipeline()
        result = pipe.analyze(str(save_path), save_report=False)

        
        result_path = RESULTS_FOLDER / f"{job_id}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        
        return jsonify({
            "job_id":       job_id,
            "status":       "success",
            "final_scores": result.get("final_scores", {}),
            "transcript":   result.get("transcript", ""),
            "full_report":  result
        })

    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {e}")
        return jsonify({"error": str(e), "job_id": job_id}), 500


@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    """
    Analyze a plain-text transcript.
    Body: { "transcript": "..." }
    """
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "Missing 'transcript' field"}), 400

    transcript = data["transcript"].strip()
    if len(transcript) < 20:
        return jsonify({"error": "Transcript too short (min 20 chars)"}), 400

    try:
        pipe   = get_pipeline()
        result = pipe.analyze_transcript(transcript)
        return jsonify({"status": "success", **result})
    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/results/<job_id>")
def get_result(job_id: str):
    """Retrieve a cached analysis result by job ID."""
    result_path = RESULTS_FOLDER / f"{job_id}.json"
    if not result_path.exists():
        return jsonify({"error": "Result not found"}), 404
    with open(result_path) as f:
        return jsonify(json.load(f))


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max size: 500MB"}), 413


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
