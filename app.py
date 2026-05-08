
from flask import Flask, request, jsonify, render_template
from src.pipeline.predict import PredictionPipeline

app      = Flask(__name__)
pipeline = PredictionPipeline()   # loaded once at startup


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(pipeline.health_check())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        result = pipeline.predict(data)
        return jsonify({
            "prediction":   result.prediction,
            "label":        result.label,
            "confidence":   result.confidence,
            "prob_leq_50k": result.prob_leq_50k,
            "prob_gt_50k":  result.prob_gt_50k,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        records = request.get_json(force=True)
        if not isinstance(records, list):
            return jsonify({"error": "Expected a JSON array of records"}), 400
        results = pipeline.predict_batch(records)
        return jsonify([{
            "prediction":   r.prediction,
            "label":        r.label,
            "confidence":   r.confidence,
            "prob_leq_50k": r.prob_leq_50k,
            "prob_gt_50k":  r.prob_gt_50k,
        } for r in results])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)