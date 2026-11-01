# Title Promise Checker + YouTube watch page

## What it does

The **Title Promise** page includes a small built-in demo shelf, an optional live YouTube Data API v3 search, and an embedded player for the chosen public YouTube video. It tries to retrieve public captions and compares the title's core expectation with timestamped transcript windows. It returns a JSON-shaped `title_promise_analysis` object with:

- `extracted_promise`
- one `fulfillment_status`: `Fulfilled`, `Partly Fulfilled`, or `Not Enough Evidence`
- an objective `reasoning` string
- up to three `evidence_timestamps` with transcript excerpts and matched terms
- optional filler/promotion excerpts
- method and coverage diagnostics
- a 0–100 **Title–Transcript Match Index** that combines title-term coverage, the strongest evidence windows, and a small filler penalty
- separate topic-term coverage and best-evidence strength scores, with a transparent note stating whether local sentence-transformer embeddings were used

A timestamp range is displayed only when the supplied transcript includes timestamps. Untimed pasted transcript text is never given invented timestamps. The page lets the user click evidence timestamps to reload the embedded player at that point.

## Windows setup

From the project root in an activated `.venv`:

```powershell
pip install -r requirements-promise.txt
```

This installs `youtube-transcript-api` for public caption retrieval. For an optional local semantic window-ranking model, additionally run `pip install -r requirements-promise-semantic.txt`. It downloads the compact `sentence-transformers/all-MiniLM-L6-v2` weights from Hugging Face on first use and runs on CPU. On a low-memory laptop, start with captions-only and skip the semantic package. If the optional semantic package/model is missing or cannot load, the checker falls back to transparent lexical evidence rules. The method used is included in every result.

Start/restart the app:

```powershell
python -m uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and choose **Title Promise**. Click either built-in sample card to load its embedded player, or paste a YouTube watch URL, `youtu.be` URL, Shorts URL, or 11-character video ID and click **Load player**. The title may be filled from YouTube's public oEmbed metadata; edit it when necessary. Search by topic can be performed inside Veri-Click after you configure the optional YouTube Data API key; direct video-link loading remains available without that key.

For live search by topic, follow [`YOUTUBE_API_SETUP.md`](YOUTUBE_API_SETUP.md) to set up a local YouTube Data API v3 key. Enter a topic under **Search YouTube inside Veri-Click** and choose a result to load it into the embedded player. Search results are query-specific and paginated; this cannot enumerate all YouTube videos.

Click **Check title promise** with the transcript box empty to try fetching public captions. Some videos have no accessible captions or YouTube may limit automated requests. In that case, paste a timestamped transcript. Supported examples:

```text
[00:12 - 00:20] The speaker explains the main definition.
00:21 The next section gives a practical example.
```

Untimed transcript text can also be checked, but the report will say timestamps are unavailable.

## Interpretation / limitations

This is a local evidence-audit aid, not a factuality verdict. The Match Index is a composite score on a 0–100 scale, **not a probability** that the title is true or fulfilled, and not a calibrated confidence value. The current implementation does not call a generative LLM. Without the optional semantic package, it uses lexical topic matching and transparent content/filler rules. With the optional sentence-transformer package, it adds local text embeddings to rank evidence windows; embeddings measure similarity, not logical entailment. A human should review the cited windows before alleging a title is misleading.

The embedded player uses the standard YouTube embedded-player URL and plays YouTube-hosted content; this project does not download YouTube video files. Embedding remains subject to YouTube's player rules and the video's owner/embed settings. For more details, see [YouTube's embed instructions](https://support.google.com/youtube/answer/171780) and the [youtube-transcript-api project](https://pypi.org/project/youtube-transcript-api/).

## PromiseProof: claim-wise title vs transcript evidence

The PromiseProof report decomposes common title shapes into individual claims and audits each against available transcript windows. It currently recognizes common topic clauses, numbered list/count promises, explicit free/no-cost claims, quantified outcomes (for example, "save 10 hours"), duration claims (for example, "in 10 minutes"), and no-code claims. Each claim gets one of `Supported`, `Partly supported`, or `No evidence found`, plus a rule-based evidence score and an excerpt/timestamp when a transcript window can be linked.

The top-level **Evidence coverage** is the percentage of extracted claims with at least partial transcript support. The report separately counts fully supported, partly supported, and no-evidence claims. If captions are short, it displays a limited-evidence warning. A claim with no transcript match is **not automatically false**; captions may be incomplete, and relevant information might appear visually or in an omitted section.

The PromiseProof method uses transparent title-claim rules and transcript-window matching. If the optional local sentence-transformer package is installed, it can help rank topic windows by semantic similarity; this is not a natural-language inference/entailment model and does not independently verify truth. No generative LLM is called. The same report includes the separate title-only clickbait-risk estimate, with the active mode labeled as a trained model or demo heuristic. Clickbait risk, evidence coverage, and factual truth are different quantities.
