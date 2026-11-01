# Worth Watching — Media Analyzer Guide

## What it does

- Accepts local video files (MP4, MOV, MKV, WebM, AVI and related formats) and audio files (MP3, WAV, M4A, AAC, OGG, FLAC, WMA).
- Uses a local Whisper speech-recognition model to transcribe the audio track.
- Provides an extractive summary selected from transcript sentences, up to five timestamped transcript windows, transcript text, detected language and duration.
- Samples up to three video frames for preview and reports simple frame brightness/contrast/saturation; it does not semantically classify every frame.
- Asks the user what they want from the video, then estimates a keyword-based match between that goal and the transcript.
- Provides a preliminary fit score: 75% goal keyword match + 25% heuristic content specificity. This is a transparent orientation aid, not a calibrated probability or objective quality/truth score.

## Windows setup

From the project root in PowerShell, with `.venv` activated:

```powershell
pip install -r requirements.txt
pip install -r requirements-media.txt
uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
```

Then visit `http://127.0.0.1:8000`, choose **Worth watching**, upload a file, and enter a goal such as `I want to learn Python basics for my exam`. The first request downloads the small multilingual Whisper `tiny` model, so internet is required for that first model download. Subsequent transcription can run locally. Default model configuration is CPU `int8` to reduce memory demand.

## Resource guidance

- Test a 1–5 minute media clip first, preferably under 100 MB on low-RAM laptops.
- The endpoint limit is 200 MB, but a larger or longer file may take a long time on CPU.
- A clear speech track is required; music-only or silent files will be rejected as not enough spoken content.
- The browser's embedded preview may not play every uploaded codec even if PyAV can decode it for analysis.

## Privacy and limitations

Media is copied to a temporary server-side file for the local analysis request and deleted when the request finishes. The transcript is returned to the browser but is not saved to SQLite history. Speech recognition is probabilistic and may make errors, particularly with noisy audio, overlapping speakers or code-switching. Goal match uses token overlap, so it may miss synonyms or conceptually related phrases. The extractive summary selects existing transcript sentences; it does not generate new facts. The “worth watching” recommendation should be treated as a preview aid, not a substitute for personal judgment, fact-checking or a calibrated quality model.
