# Train and compare four deep-learning models

This experiment trains four models using the **same stratified train/validation/test split**:

1. **BiGRU text-only** — title token sequence.
2. **CNN thumbnail-only** — image pixels at 96×96.
3. **MLP engineered-feature baseline** — 16 lexical/thumbnail-style features.
4. **Multimodal gated fusion** — BiGRU + CNN + feature MLP, the primary model deployed by Veri-Click.

All models use binary labels: `1 = clickbait`, `0 = non-clickbait`. The comparison table reports held-out test accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and elapsed training time. The best epoch is selected using validation F1, not test data. The random seed and split counts are written to `artifacts/model_comparison_metrics.json`.

## Recommended low-resource Windows pilot

1. Download `Final_data.csv` from the YTClickbait21K Figshare page and save it as `data/Final_data.csv`.
2. Activate the existing `.venv` and install DL dependencies:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-dl.txt
   ```

3. Run the small pilot (only the sampled rows' thumbnail URLs are downloaded, and only from allow-listed YouTube image hosts):

   ```powershell
   python scripts/compare_models.py --csv data/Final_data.csv --image-root data --download-thumbnails --max-samples 300 --epochs 3 --batch-size 4
   ```

   The script's `--download-thumbnails` option is optional if you have already extracted matching local thumbnails. It does **not** need the entire 2.41 GB archive for this pilot. A working internet connection is needed for thumbnail downloads when local images are missing. The data CSV and images stay in the project folder.

4. After the command finishes, open **Model Lab** or refresh that page. It should show the experiment table. Restart the API in its terminal to load the trained fusion model for title/thumbnail inference.

For a larger run, increase `--max-samples` gradually. On an older CPU-only laptop, 300 examples × 3 epochs is a pilot; expect runtime to depend on CPU, disk, and image-download speed. Avoid starting with the entire dataset. To train on all valid rows, set `--max-samples 0` only after the pilot succeeds.

## Outputs

- `artifacts/model_comparison_metrics.json` — four models' test results, split info, class counts, and training times.
- `artifacts/model_comparison.csv` — easy-to-open metric table.
- `artifacts/model_comparison_history.csv` — train/validation loss and F1 per epoch.
- `artifacts/comparison_models/*.pt` — checkpoint for each model.
- `artifacts/multimodal_model.pt`, `vocab.json`, `model_config.json`, `test_metrics.json` — fusion checkpoint and the existing application inference artifacts.
- `artifacts/comparison_models/prepared_dataset_used.csv` — exact title/label/image rows used.

## Research-quality cautions

- A pilot can be noisy; report the dataset, sample size, split, class counts, seed and all metrics.
- The metrics are from the experiment run, not the values reported in the supplied paper. This implementation does not reproduce the paper's RoBERTa + ResNet50 + PCA + stacked ensemble.
- The `Feature MLP` is an engineered-feature baseline; the CNN is image-only; and the fusion model is the proposed multimodal model. Do not call all branches separate pretrained models.
- This script evaluates clickbait labels, not whether a video's claims are factually true or whether it will be personally enjoyable.
