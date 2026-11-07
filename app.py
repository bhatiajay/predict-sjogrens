"""Web app: enter gene expression values, get the model's score.

    uv run flask --app app run --port 5001

(Port 5000 is used by AirPlay Receiver on macOS, so this uses 5001.)

The app only loads a model that train.py already saved. It never trains, so it
starts in seconds and always serves the same model.
"""

import math
from pathlib import Path

from flask import Flask, render_template, request

from sjogrens import artifacts, config
from sjogrens.model import predict_scores


def create_app(model_dir: Path | str = config.MODEL_DIR) -> Flask:
    app = Flask(__name__)
    loaded = artifacts.load(model_dir)
    model, meta = loaded if loaded else (None, None)

    def missing_model():
        return render_template("missing_model.html"), 503

    @app.get("/")
    def index():
        if model is None:
            return missing_model()
        example = request.args.get("example")
        values = meta["examples"].get(example, {}) if example else {}
        return render_template("index.html", meta=meta, values=values, errors={})

    @app.post("/predict")
    def predict():
        if model is None:
            return missing_model()

        values, errors = {}, {}
        for gene in meta["genes"]:
            raw = request.form.get(gene, "").strip()
            try:
                value = float(raw)
            except ValueError:
                errors[gene] = "Required." if not raw else f"'{raw}' isn't a number."
                continue
            if not math.isfinite(value):
                errors[gene] = "Must be a finite number."
                continue
            values[gene] = value

        if errors:
            return render_template("index.html", meta=meta, values=request.form, errors=errors), 400

        row = [[values[g] for g in meta["genes"]]]
        score = float(predict_scores(model, row)[0])
        out_of_range = [
            (g, values[g], *meta["ranges"][g])
            for g in meta["genes"]
            if not meta["ranges"][g][0] <= values[g] <= meta["ranges"][g][1]
        ]
        return render_template(
            "result.html", meta=meta, values=values, score=score, out_of_range=out_of_range
        )

    return app


if __name__ == "__main__":
    create_app().run(port=5001)
