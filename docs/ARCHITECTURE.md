# System Architecture

```text
Browser UI (static HTML / CSS / JavaScript)
      | JSON over localhost
      v
FastAPI local service
  |-- /api/analyze --------> analyzer.py
  |                           |-- trained checkpoint present? -> BiGRU + CNN gated-fusion model
  |                           |-- otherwise -> transparent demo risk index
  |                           |-- title signals + thumbnail style statistics
  |-- /api/media-analyze --> media_analyzer.py
  |                           |-- PyAV decodes audio/video metadata and samples up to 3 video frames
  |                           |-- faster-whisper tiny CPU/int8 transcribes spoken audio
  |                           |-- extractive transcript summary + timestamp windows
  |                           |-- keyword overlap against user viewing goal + transparent heuristics
  |                           |-- temporary upload deleted after request; transcript not saved
  |-- /api/media-capabilities -> optional speech dependency/model status
  |-- /api/youtube/metadata -> validated public YouTube oEmbed title / embed URL
  |-- /api/youtube/promise-check -> public captions or pasted transcript -> title-promise evidence/status
  |-- /api/batch ----------> analyze up to 200 titles
  |-- /api/history --------> local SQLite database
  |-- /api/dashboard ------> aggregate counts and average risk index
  |-- /api/model-details --> checkpoint and held-out metric metadata

Chrome Manifest V3 content script
      | title + allow-listed YouTube thumbnail URL
      v
FastAPI /api/analyze (save_history=false)
      |
      v
badge injected on up to 12 visible YouTube video cards
```

## Model architecture

- **Text branch:** tokenization, train-split vocabulary, Embedding, bidirectional GRU, masked sequence pooling, projection to a 128-dimensional text vector.
- **Image branch:** 96 × 96 RGB input, compact four-block CNN, adaptive average pooling and projection to 128 dimensions.
- **Side features:** 16 fixed signals from lexical cues and basic image-style statistics, projected to 32 dimensions.
- **Fusion:** learned sigmoid gate combines the text and image vectors dimension-wise. Side features are appended after fusion.
- **Head:** MLP (160 → 96 → 32 → 1) with sigmoid at inference.

The included compact network is trained from scratch and is not a pretrained RoBERTa/ResNet model. Its generalization depends on a real, correctly labeled dataset. No random untrained weights are used for user-facing classification.

## API

- `GET /api/health`: local service and current inference mode.
- `GET /api/dashboard`: saved-history aggregate statistics.
- `GET /api/history?limit=50`: recent analyses.
- `DELETE /api/history`: clear local analysis history.
- `GET /api/demo-cases`: three illustrative demo inputs.
- `GET /api/model-details`: training artifact availability and stored test metrics.
- `POST /api/analyze`: `{ "title": "...", "image_data": "data:image/jpeg;base64,...", "thumbnail_url": null, "save_history": true }`.
- `POST /api/batch`: `{ "items": [{"title":"..."}] }`, up to 200 items.
- `GET /api/media-capabilities`: optional `faster-whisper` availability and configured model size.
- `POST /api/media-analyze`: multipart fields `file`, optional `title`, and `goal`; analyzes a supported uploaded audio/video file. Limits uploads to 200 MB and rejects unsupported suffixes.

## Privacy and safety controls

The service binds to loopback in the documented start command. Upload images are limited to 5 MB and validated with Pillow. Remote image fetches accept HTTPS URLs only on `ytimg.com`, `youtube.com` or `googleusercontent.com` host families, reject non-default ports and credentials, disable redirects, set request timeouts and stop reading after 5 MB. CORS is restricted to the local dashboard origins and YouTube origins for the content script. History is local SQLite and can be cleared using the API. The Chrome extension does not save each badge check to history.

## Worth Watching media workflow

`vericlick/media_analyzer.py` is distinct from the title/thumbnail clickbait classifier. Speech recognition is done by `faster-whisper` using the multilingual `tiny` model on CPU/int8 by default. PyAV (installed as a dependency of faster-whisper) decodes the file, reports duration, and samples up to three frames. The report provides a transcript preview, extractive sentence summary, timestamped transcript windows, language metadata, keyword overlap with the user's goal, and a heuristic content-specificity score.

The combined “preliminary fit score” is `0.75 × goal keyword match + 0.25 × content specificity estimate`, when a goal is provided. These components are transparent heuristics, not calibrated probabilities and not evidence that the content is true or objectively high-quality. Frame sampling calculates simple brightness/contrast/saturation signals; it does not use a semantic vision model. Uploaded media is stored temporarily for processing and deleted in a `finally` block; media results are not written to the SQLite history.

## Title Promise workflow

The browser page embeds a selected YouTube video via the standard privacy-enhanced YouTube iframe host. The API never downloads the video file. It fetches public captions through the optional `youtube-transcript-api` package or parses a user-pasted transcript. `vericlick/title_promise.py` deconstructs common title forms into a promise kind, groups transcript snippets into timestamp windows, detects possible filler/promotions, ranks evidence, and returns `Fulfilled`, `Partly Fulfilled`, or `Not Enough Evidence`. If `sentence-transformers` is installed, `all-MiniLM-L6-v2` can add local semantic-similarity ranking; otherwise it falls back to lexical coverage. Embedding similarity is not entailment and all statuses remain provisional transcript evidence judgments—not factuality classifications.
