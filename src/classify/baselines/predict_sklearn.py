# -----------------------------------------------------------------------------
# Inference for the TF-IDF + Logistic Regression baseline
#
# Usage:
#   python src/classify/baselines/predict_sklearn.py models/cls_tfidf_logreg "your note text"
#
# Output JSON (mirrors your HF predictor):
# {
#   "label": "Gastroenterology",
#   "probs": {"Cardiovascular / Pulmonary": 0.07, "Gastroenterology": 0.52, ...}
# }
# -----------------------------------------------------------------------------

import sys, json, joblib, numpy as np


def softmax(z):
    """Stable softmax for fallback when predict_proba is unavailable."""
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main():
    if len(sys.argv) < 3:
        print("Usage: python src/classify/baselines/predict_sklearn.py <model_dir> \"<note text>\"")
        sys.exit(1)

    model_dir = sys.argv[1]
    text = sys.argv[2]

    # Load pipeline + config
    pipe = joblib.load(f"{model_dir}/model.joblib")
    cfg = json.load(open(f"{model_dir}/config.json"))
    id2label = {int(k): v for k, v in cfg["id2label"].items()}  # keys are strings in JSON

    # Prefer predict_proba (LogReg supports it). Fallback: decision_function -> softmax.
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        probs = pipe.predict_proba([text])[0].tolist()
    else:
        scores = pipe.decision_function([text])[0]
        if np.ndim(scores) == 0:  # binary corner-case
            scores = np.array([scores, -scores])
        probs = softmax(scores).tolist()

    labels = [id2label[i] for i in range(len(probs))]
    pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

    out = {
        "label": pairs[0][0],
        "probs": {k: float(v) for k, v in pairs}
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
