"""Evaluate the model with cross-validation, then train the final model and save it.

    uv run python train.py                   # 5 report genes, full evaluation (~2 min)
    uv run python train.py --features all    # all 13 genes, for comparison
    uv run python train.py --skip-cv         # just train and save the final model

Output: models/model.keras (the network, including its scaling layer) and
models/meta.json (gene list, input ranges, example values, CV results), which the web
app loads.
"""

import argparse
import time

import sjogrens  # noqa: F401
import tensorflow as tf

from sjogrens import artifacts, config
from sjogrens.data import load_dataset
from sjogrens.evaluate import cross_validate, format_summary, summarize
from sjogrens.model import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=config.FEATURE_SETS, default="report")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-cv", action="store_true", help="skip evaluation, only train the final model")
    parser.add_argument("--out", default=str(config.MODEL_DIR), help="where to save the model")
    args = parser.parse_args()

    # Same seed + deterministic ops = the same numbers on every run.
    tf.config.experimental.enable_op_determinism()

    genes = config.FEATURE_SETS[args.features]
    X, y = load_dataset(genes=genes)
    print(f"Data: {len(y)} samples ({(y == 1).sum()} patients, {(y == 0).sum()} controls)")
    print(f"Genes ({len(genes)}): {', '.join(genes)}\n")

    cv_summary = None
    if not args.skip_cv:
        print(f"Cross-validation: {args.splits} folds x {args.repeats} repeats, {args.epochs} epochs each")
        start = time.time()
        results = cross_validate(X, y, n_splits=args.splits, n_repeats=args.repeats, epochs=args.epochs)
        summary = summarize(results)
        print(f"\nResults across {len(results) // 3} test folds (mean ± std), {time.time() - start:.0f}s:\n")
        print(format_summary(summary), "\n")
        cv_summary = {
            "splits": args.splits,
            "repeats": args.repeats,
            "epochs": args.epochs,
            "results": summary.reset_index().to_dict(orient="records"),
        }

    # Cross-validation estimated how well this recipe works. The model we keep is a
    # fresh one trained the same way on all 160 samples; picking the "best fold" model
    # would be choosing by luck and would waste 20% of the data.
    print("Training final model on all samples...")
    model = train(X, y, epochs=args.epochs)
    out = artifacts.save(model, artifacts.build_meta(X, y, genes, cv_summary), args.out)
    print(f"Saved {out / config.MODEL_FILE} and {out / config.META_FILE}")


if __name__ == "__main__":
    main()
