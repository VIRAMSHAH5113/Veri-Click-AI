# Trained model artifacts

This directory intentionally contains no trained model weights or fabricated benchmark scores at delivery. The supplied PDFs do not include a matching labeled dataset or a trained checkpoint.

## Four-model comparison (recommended)

Use `scripts/compare_models.py` to train four comparable models on the same split. See [`docs/MODEL_COMPARISON_GUIDE.md`](../docs/MODEL_COMPARISON_GUIDE.md) for the YTClickbait21K workflow. The script saves:

- `model_comparison_metrics.json` and `model_comparison.csv` — actual held-out test results.
- `model_comparison_history.csv` — train/validation history.
- `comparison_models/` — BiGRU, CNN, feature-MLP and fusion checkpoints plus the exact prepared rows.
- `multimodal_model.pt`, `vocab.json`, `model_config.json`, and `test_metrics.json` — the fusion checkpoint and metadata used by the running app.

For a single fusion model only, use `scripts/train_model.py`. The API activates the main checkpoint only when its model, vocabulary and config artifacts are present.
