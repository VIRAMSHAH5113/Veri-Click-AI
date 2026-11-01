# Veri-Click AI — Multimodal Clickbait Risk Screening

**A local-first deep-learning research prototype and Chrome companion for YouTube.** It expands the supplied Veri-Click idea into a usable dashboard, API, transparent demo mode, optional trainable multimodal neural network, batch audit workflow, history, browser overlay, and automated tests.

> **Research honesty:** the supplied paper reports 97.34% accuracy / 0.9728 F1 for its RoBERTa + ResNet50 + handcrafted features + PCA + stacked ensemble pipeline. This ZIP does **not** claim to reproduce those numbers. The PDFs do not include the exact labeled 37,397-row thumbnail dataset or trained model checkpoint. Without those artifacts, the included dashboard starts in clearly labelled **demo heuristic** mode. You can train the included compact neural network on your own labeled title + thumbnail dataset; its held-out metrics will be stored locally and displayed in Model Lab.

## What is included

- **Modern dashboard:** single-title analysis, optional thumbnail upload, risk signal breakdown, recent history, CSV batch screening/export, model status and architecture guide.
- **Worth Watching media audit:** upload a local video/audio file, preview it, transcribe speech with Whisper, show an extractive summary and timestamped transcript windows, sample up to three video frames, estimate relevance against a goal you provide, and include a title-promise check when a title is entered.
- **Title Promise + PromiseProof:** embed a selected YouTube video inside the app, retrieve public captions when available, or accept a pasted timestamped transcript. The original title-promise status (`Fulfilled`, `Partly Fulfilled`, `Not Enough Evidence`) is paired with a claim-wise audit (`Supported`, `Partly supported`, `No evidence found`), per-claim timestamps, an evidence-coverage percentage, limited-transcript warnings, the separate title-only clickbait-risk estimate, and downloadable JSON. Evidence coverage is not a truth/accuracy score.
- **FastAPI backend:** `/api/health`, `/api/analyze`, `/api/media-analyze`, `/api/media-capabilities`, `/api/youtube/metadata`, `/api/youtube/search-status`, `/api/youtube/search`, `/api/youtube/promise-check`, `/api/batch`, `/api/history`, `/api/dashboard`, `/api/model-details`, `/api/demo-cases`.
- **Chrome extension (Manifest V3):** user-triggered scan of up to 12 visible YouTube cards with local risk badges. It does not scrape or upload comments/user profiles and does not claim to fact-check video truth.
- **Deep-learning model ready to train:** bidirectional GRU for title sequences + compact CNN for thumbnails + 16 numeric side signals; a learned gate fuses the text/image representations, then an MLP produces a binary logit.
- **Training/evaluation:** stratified train/validation/test split (70/15/15 approximately), vocabulary created from the training split, class-weighted binary loss, AdamW, gradient clipping, early stopping on validation F1, checkpoint selection by validation F1, and held-out test metrics including confusion matrix and ROC-AUC.
- **Tests:** feature extraction, API validation, image upload, batch analysis, SQLite history, model forward pass, and training data/metrics helpers.
- **Research documentation:** a source-grounded review of the paper, report and slides; architecture, limitations, reproducibility gap, roadmap and literature links in `docs/`.

## Quick start (Windows 11)

Use **Python 3.11 or 3.12** for the easiest dependency installation. Open PowerShell in the extracted `Veri-Click-AI` folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in Chrome or Edge. Leave the terminal open while using the app. For Windows PowerShell execution-policy errors, use Command Prompt and activate with `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe -m uvicorn vericlick.api:app --host 127.0.0.1 --port 8000` after installing dependencies in that environment.

### Enable video/audio content analysis

The original dashboard works without a downloaded AI speech model. To enable **Worth watching**, stop the server with `Ctrl+C`, keep `.venv` activated, and run:

```powershell
pip install -r requirements-media.txt
uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
```

The first media analysis downloads the small multilingual **Whisper `tiny`** speech model (internet required only for that first model download). It runs on CPU with int8 mode to lower memory use. On a low-RAM laptop, begin with a 1–5 minute clip and a file below 200 MB. You can change the model using `VERICLICK_WHISPER_MODEL`, but larger models need more time and memory. The page accepts local media files; it does not download videos directly from YouTube.

### Enable YouTube Title Promise Checker

From the same activated `.venv`, run:

```powershell
pip install -r requirements-promise.txt
```

This installs public YouTube caption retrieval. To add optional local sentence-transformer similarity ranking, run `pip install -r requirements-promise-semantic.txt`; this downloads additional model weights and uses extra RAM, so skip it on a low-memory laptop. Without it, the feature still works using transparent lexical evidence rules and discloses its method. Go to **Title Promise**, choose a video or paste a YouTube URL/video ID, click **Load player**, then **Check title promise**. The report includes the original fulfilment status and **PromiseProof**, which splits common title patterns into checkable claims, shows `Supported` / `Partly supported` / `No evidence found` for each, links transcript evidence to timestamps, and calculates the share of claims with at least partial transcript evidence. If captions cannot be retrieved, paste a timestamped transcript. PromiseProof does not independently fact-check the video and does not use a generative LLM. See [`docs/TITLE_PROMISE_YOUTUBE_GUIDE.md`](docs/TITLE_PROMISE_YOUTUBE_GUIDE.md).

### Enable in-app YouTube search (optional API key)

Copy `.env.example` to `.env`, create a key in the [Google Cloud Console](https://console.cloud.google.com/) after enabling **YouTube Data API v3**, restrict the key to that API, and set `YOUTUBE_API_KEY=your_actual_key` in `.env`. Keep the key local; do not paste it into the browser or chat, and do not commit `.env` to GitHub. Restart the app, then use **Title Promise → Search YouTube inside Veri-Click**. This returns query-specific, paginated results (not every YouTube video); API quota applies. Full instructions: [`docs/YOUTUBE_API_SETUP.md`](docs/YOUTUBE_API_SETUP.md). The direct video-link loader and sample shelf still work without a key.

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
```

No external model download or API key is needed for the first launch. The sample thumbnails are synthetic demo art and must not be treated as training evidence.

## How to use the dashboard

1. **Overview:** see local model status and recent analyses.
2. **Analyze video:** enter a YouTube title and optionally upload a thumbnail. Try the built-in examples first.
3. **Worth watching:** choose a local video/audio file and write a goal such as “learn Python basics for my exam”. The page transcribes speech, extracts a short summary and timestamped windows, samples up to three frames, and returns a preliminary goal-fit recommendation. If you also enter the video title, it shows the Title Promise assessment when transcript evidence is available.
4. **Title Promise / PromiseProof:** select a video from in-app YouTube search (or paste a URL), load the embedded player, and audit the title against public captions or a pasted transcript. Review each title claim and linked evidence, compare the claim-wise evidence coverage with the separate clickbait-risk estimate, and download the JSON report. Neither output verifies factual truth; caption coverage may be incomplete.
5. **Batch audit:** upload a CSV with a title column named `video_title`, `title`, `video title`, or `headline`. Use **Download sample CSV** to get a compatible example. The CSV uploader accepts up to 200 title rows and 2 MB.
6. **Model lab:** inspect the clickbait model mode and saved training artifacts.

### Understanding the score

The three bands are a **screening index**: High risk (`>=62`), Review (`35–61`), Likely informative (`<35`). In default demo mode, the score is made from transparent lexical rules plus weak thumbnail-style statistics (brightness, saturation, contrast, edges, etc.). These visual statistics **do not understand what an image depicts** and cannot prove that a title is misleading. The score is not a calibrated probability, a truth verdict, or a substitute for human review. In trained mode, the binary neural model predicts the learned dataset label; that output is still dataset-dependent and should be reported with the held-out metrics.

The **Worth Watching** page is a separate content utility. Whisper is the neural speech-recognition model; the short summary is extractive (selected from transcript sentences), while goal match is keyword overlap and content specificity is a heuristic. This is useful for previewing relevance, but it is not a factuality verifier, a general-purpose video-understanding foundation model, or a calibrated “worth it” probability. Up to three sampled frames are shown with basic visual statistics; the app does not semantically understand every frame.

## Train and compare the deep-learning models

The recommended experiment trains four CPU-friendly models on the same stratified train/validation/test split:

1. **BiGRU text-only** — title sequence model.
2. **CNN thumbnail-only** — image-pixel model.
3. **MLP engineered-feature baseline** — 16 lexical/thumbnail-style signals.
4. **Multimodal gated fusion** — combines BiGRU, CNN and side-feature representations; this is the primary deployed model.

Use a labeled CSV with title, `label` or `final_label` (1=clickbait, 0=non-clickbait), and local-thumbnail paths or a `thumbnail` URL. The YTClickbait21K `Final_data.csv` schema is supported (`title`, `final_label`, `thumbnail`, `local_thumbnail`).

### Recommended Windows CPU pilot

1. Download `Final_data.csv` from the YTClickbait21K Figshare page and save it as `data/Final_data.csv`.
2. Activate `.venv`, then install the DL dependencies:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-dl.txt
   ```

3. Run a small pilot. This downloads only sampled thumbnail images from allow-listed YouTube image hosts when local files are absent. You do **not** need to download the whole 2.41 GB thumbnail archive to start:

   ```powershell
   python scripts/compare_models.py --csv data/Final_data.csv --image-root data --download-thumbnails --max-samples 300 --epochs 3 --batch-size 4
   ```

   On an older CPU-only laptop, this is a pilot and may take a while. Internet access is needed only for images that are not already present locally. Increase the sample size gradually after the pilot completes. Use `--max-samples 0` only when you are ready to train with every valid row.

4. Restart the API so the trained fusion checkpoint becomes active, then open **Model Lab**. The model comparison table is filled only after a real run; no example scores are pre-populated.

### Metrics and artifacts

All four models use the same held-out test rows. The script reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix and elapsed time. The best epoch is selected using validation F1; the test set is not used for epoch selection.

Outputs in `artifacts/`:

- `model_comparison_metrics.json` — per-model test metrics, sample counts, class counts and split seed.
- `model_comparison.csv` — comparison table suitable for Excel/reporting.
- `model_comparison_history.csv` — train/validation loss and F1 by epoch.
- `comparison_models/*.pt` — four individual model checkpoints.
- `multimodal_model.pt`, `vocab.json`, `model_config.json`, `test_metrics.json` — fusion checkpoint and files used by the live app.
- `comparison_models/prepared_dataset_used.csv` — exact image-backed rows used in the run.

The original single-model trainer is still available for a focused fusion-only run:

```powershell
python scripts/train_model.py --csv data/my_labeled_videos.csv --image-root data --max-samples 500 --epochs 3 --batch-size 4
```

### Research-quality cautions

- Metrics from a 300-row pilot can vary substantially. Report dataset, sample size, class counts, train/validation/test split, seed and all test metrics.
- These models do not reproduce the supplied paper's RoBERTa + ResNet50 + PCA + stacked-ensemble pipeline. Never claim the paper's 97.34% accuracy as a result from this implementation.
- The CNN learns visual patterns but is not a general-purpose semantic video-understanding model. The feature MLP is an engineered-feature baseline. Whisper is used separately for audio transcription on the **Worth watching** page.
- Clickbait labels are not factuality labels. A model prediction does not prove that a video's claims are false or that watching it is/isn't personally worthwhile.

See [`docs/MODEL_COMPARISON_GUIDE.md`](docs/MODEL_COMPARISON_GUIDE.md) for detailed setup, expected artifacts and experiment limitations.

## Install the Chrome extension

1. Start the local API using the Quick start steps.
2. In Chrome, open `chrome://extensions` and enable **Developer mode**.
3. Click **Load unpacked** and choose the `extension/` directory inside the extracted project.
4. Open YouTube and click **Scan visible videos** in the lower-right corner.
5. The extension asks the local API to screen at most 12 visible cards and places small risk badges over their thumbnails.

The extension sends title text and a YouTube thumbnail URL to your local service. It does not use an external AI service. The API accepts HTTPS image URLs only from YouTube/Google thumbnail host suffixes and limits the image size. Badges are screening cues, not factual verification. The extension source is included; it has not been published to the Chrome Web Store.

## Test suite and checks

```powershell
pytest -q
python -m compileall vericlick scripts tests
node --check static/app.js
node --check extension/content.js
# Optional training CLI check, after installing requirements-dl.txt:
python scripts/train_model.py --help
```

The tests do not need internet access, pretrained checkpoints, or the full dataset. Neural-network and training tests run only when PyTorch is installed. For the exact pass count and manual test boundaries, see `docs/TEST_REPORT.md`; browser-level QA on the target Chrome/YouTube layout remains necessary.

## API examples

```powershell
# Health/model status
curl http://127.0.0.1:8000/api/health

# Single title (do not store in history)
curl -X POST http://127.0.0.1:8000/api/analyze `
  -H "Content-Type: application/json" `
  -d '{"title":"How to learn linear algebra: a complete guide","save_history":false}'

# Batch audit
curl -X POST http://127.0.0.1:8000/api/batch `
  -H "Content-Type: application/json" `
  -d '{"items":[{"title":"STOP! You will not believe this secret"},{"title":"How to replace a light switch safely"}]}'
```

## Local data and privacy

All analysis/history are local by default and the SQLite database is `data/history.sqlite3`. Set `VERICLICK_DB` to point at another location. The optional browser extension calls `127.0.0.1` only. For sensitive titles, use `save_history=false` or clear history from the dashboard/API. A remote thumbnail URL is fetched server-side only after URL allow-list validation; use direct upload for an image you already have. The media endpoint processes uploaded audio/video in a temporary file, then deletes it; it does not save media or transcript to SQLite history.

## Project layout

```text
vericlick/              Feature extraction, analyzers, optional DL model, media/promise pipelines, API, SQLite storage
static/                 Dashboard HTML/CSS/JS and synthetic demo thumbnails
extension/              Chrome Manifest V3 extension source
scripts/train_model.py  Train, validate, test and save the multimodal DL model
scripts/compare_models.py Train and compare four DL model configurations
tests/                  Unit, integration and API tests
requirements-media.txt  Optional faster-whisper dependency for audio/video analysis
requirements-promise.txt Optional YouTube public-caption retrieval for Title Promise
requirements-promise-semantic.txt Optional sentence-transformer semantic ranking (extra RAM/model download)
docs/                   Research review, architecture, media/promise guides and roadmap
data/demo_cases.json    Built-in demo examples
artifacts/              Empty until you train; no checkpoint is fabricated
outputs/                Reserved for optional exports
```

## Source basis

The system is grounded in the three supplied files: the paper *Multimodal Clickbait Detection in YouTube Videos using RoBERTa and Stacked Ensemble Learning*, the `Final 0.4` project/presentation PDF, and the 11-slide `Veri-Click: Truth in Headlines` deck. Original pipeline and reported metrics are documented in `docs/RESEARCH_REVIEW.md`; implementation trade-offs and research improvements are explained in `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`. `docs/DEMO_AND_VIVA_GUIDE.md` contains a five-minute demo flow and viva preparation, and `docs/TEST_REPORT.md` lists actual test results and remaining manual checks.

## In-app YouTube search (optional API key)

See [`docs/YOUTUBE_API_SETUP.md`](docs/YOUTUBE_API_SETUP.md) for configuring YouTube Data API v3 for the Title Promise page.
