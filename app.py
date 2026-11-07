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
from sjogrens.genes import GENE_INFO, ncbi_url
from sjogrens.model import predict_scores


def _to_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def create_app(model_dir: Path | str = config.MODEL_DIR) -> Flask:
    app = Flask(__name__)
    loaded = artifacts.load(model_dir)
    model, meta = loaded if loaded else (None, None)

    # Display-only helpers for the templates. None of these change the model's score.
    reference = {}
    if model is not None:
        # Scores of the median patient and median control, shown as landmarks on the result scale.
        reference = {
            group: float(predict_scores(model, [[meta["examples"][group][g] for g in meta["genes"]]])[0])
            for group in ("control", "patient")
        }

    def position(gene: str, value) -> float | None:
        """Where a value sits on the gene's training range, as 0-100 (clamped), or None."""
        number = _to_float(value)
        if number is None:
            return None
        lo, hi = meta["ranges"][gene]
        return max(0.0, min(100.0, (number - lo) / (hi - lo) * 100))

    def in_range(gene: str, value) -> bool:
        number = _to_float(value)
        lo, hi = meta["ranges"][gene]
        return number is not None and lo <= number <= hi

    app.jinja_env.globals.update(
        gene_info=GENE_INFO, ncbi_url=ncbi_url, position=position, in_range=in_range, reference=reference
    )

    def missing_model():
        return render_template("missing_model.html"), 503

    @app.get("/")
    def index():
        if model is None:
            return missing_model()
        example = request.args.get("example")
        if example in meta["examples"]:
            values = meta["examples"][example]
        else:
            # Prefill from ?IRF5=...&SSB=... (used by "Adjust these values" on the result page).
            values = {g: request.args[g] for g in meta["genes"] if g in request.args}
        return render_template("index.html", meta=meta, values=values, errors={}, example=example)

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
            return render_template(
                "index.html", meta=meta, values=request.form, errors=errors, example=None
            ), 400

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
