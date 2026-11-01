# Verification and Test Report

**Verification date:** 26 October 2026  
**Project version:** 1.5.0 (PromiseProof claim-wise title evidence audit)

## Automated suite

Command: `pytest -o addopts='' -q --tb=short`

**Latest result: 64 passed** in the build environment, including PromiseProof claim-decomposition/evidence tests, four-model comparison smoke test, Title Promise/API tests, and mocked in-app YouTube Data API search tests.

| Area | Tests | Coverage |
|---|---:|---|
| Feature/unit tests | 5 | Sensational and informative markers, 16-feature vector, image statistics, bounded score and explanations, comparative title cues |
| Media analysis unit/integration tests | 6 | Extractive summary, context matching (Latin and Devanagari tokens), heuristic limits, fake-transcriber report generation |
| API/integration tests | 16 | Existing API checks plus media capabilities, unsupported media rejection, temporary media upload and deletion |
| Neural model tests | 3 | Vocabulary/token encoding, model forward pass and finite logits, image tensor shape/range |
| Training helper tests | 3 | CSV schema normalization, invalid-row filtering, test metric outputs, missing column errors |
| End-to-end training smoke test | 1 | Creates 40 temporary labeled rows + images, runs one-epoch fusion training, checks saved artifacts, loads checkpoint and runs inference |
| Four-model comparison tests | 3 | Public CSV schema normalization, thumbnail-host allow-list, one-epoch four-model training and saved fusion inference smoke test |
| Title Promise unit tests | 12 | YouTube URL validation, timestamp parsing, title promise status, claim decomposition for list/free/quantified/duration promises, qualification/denial handling, vague title handling and evidence coverage boundaries |
| YouTube Promise API tests | 9 | Metadata route validation, embedded URL construction, manually pasted transcript, caption fallback, missing-caption messaging, local key status, configured search response and missing-key handling |
| YouTube search adapter tests | 4 | Result normalization, pagination tokens, query/order validation, and quota-error handling using mocked API responses |
| Frontend/extension asset contract tests | 2 | Required UI IDs and unique HTML IDs; Chrome Manifest V3 assets/host configuration |

The neural-network and training tests are marked to skip when the optional PyTorch dependency is not installed. Install `requirements-dl.txt` to run the full deep-learning checks. Title Promise caption retrieval is listed in `requirements-promise.txt`; optional sentence-transformer semantic ranking is listed separately in `requirements-promise-semantic.txt`.

## Additional runtime/system checks

The app was run through a live Uvicorn server on loopback with a temporary SQLite database. The following passed:

- `/` returned HTTP 200 and the dashboard HTML.
- `/static/samples/tutorial.jpg` returned HTTP 200 with an image content type.
- `/api/health`, `/api/demo-cases` and `/api/model-details` returned HTTP 200; three demo cases were available.
- `/api/analyze` accepted a generated PNG + title, returned an in-range score and `image_available=true`.
- `/api/batch` processed two titles and returned a two-row summary.
- SQLite history and dashboard counts reflected a saved single analysis; `DELETE /api/history` removed it and subsequent history was empty.
- A thumbnail URL on an unapproved host was rejected with HTTP 400 before fetching.
- JavaScript syntax validation passed for `static/app.js` and `extension/content.js` using `node --check`.
- Python source compilation passed (`python -m compileall -q vericlick scripts tests`).
- `extension/manifest.json` parsed as valid JSON.
- `python scripts/train_model.py --help` succeeded in the build environment with PyTorch installed.
- `GET /api/media-capabilities` was tested; the upload API was tested with a stub transcriber to validate multipart handling, temporary-file cleanup and no-history persistence.
- Title Promise is tested with mocked caption/metadata calls and pasted timestamped transcripts. The optional in-app YouTube Data API search is tested with mocked API payloads; no live Google API request was made because a user-specific API key is not bundled in the project. Actual external YouTube caption availability varies by video, account/IP and network; live external caption retrieval was not claimed as a test pass.
- Optional sentence-transformer embeddings were not downloaded in the build environment. The checker uses lexical evidence rules when the optional semantic package/model is unavailable and reports the method used.
- A live Whisper transcription was not run in the build environment because the optional `faster-whisper` package and model weights were not installed there. On first use, the target machine needs `pip install -r requirements-media.txt` and an internet connection to download the tiny model.

## Important boundary

An interactive browser automation attempt was blocked by the build environment's Chromium policy (`ERR_BLOCKED_BY_ADMINISTRATOR`) for both loopback and local-file navigation. It is **not reported as a pass**. The dashboard JavaScript syntax, HTML/asset contract, backend integration, and project-side model-train-to-inference pipeline were tested, but a human visual pass in the user's Chrome browser and a live YouTube extension pass remain to be performed on the target machine.

The end-to-end model test uses synthetic data generated temporarily during the test. It proves that training artifacts can be created, reloaded and used for inference; it **does not establish model accuracy or generalization**. No synthetic-trained weights or synthetic benchmark metrics are included in the delivered ZIP.


## Four-model experiment pipeline

`tests/test_model_comparison.py` validates the public dataset schema (`title`, `final_label`, `local_thumbnail`, and thumbnail URL), the thumbnail-host allow-list, and a one-epoch synthetic smoke test that runs all four model paths, writes comparison artifacts, and loads the saved fusion checkpoint. The synthetic fixture verifies software wiring only; its metrics are not research results. A real-data run using the user's labeled dataset is still required before reporting project performance.

## PromiseProof claim-wise evidence audit

The latest test suite includes regression coverage for the title “5 free AI tools that save 10 hours”: the title is decomposed into topic, list count, free/no-cost and quantified-outcome claims; a pricing caveat makes the free claim only partly supported; an explicit denial of the promised saving prevents mention alone from counting as support; and evidence coverage is reported as claims with at least partial transcript support. These tests use synthetic transcripts and verify software behavior, not the factual accuracy of any real video.
