# In-app YouTube search setup

Veri-Click uses the official YouTube Data API v3 for query-based public video discovery. It does **not** download or mirror YouTube videos; selected videos play through YouTube's embedded player. Search result cards are kept in the current page/session and are not written to local history.

## Configure once

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Enable **YouTube Data API v3** for that project.
4. In **APIs & Services → Credentials**, create an API key.
5. Restrict the key to **YouTube Data API v3** under API restrictions. Keep the key local; do not post it in chat or commit it to GitHub.
6. In the project root, make a copy of `.env.example` named `.env`.
7. Open `.env` and replace the placeholder: `YOUTUBE_API_KEY=your_actual_key_here`. Do not add quotes unless necessary.
8. Restart the Veri-Click server.

On PowerShell, from the `Veri-Click-AI` project folder:

```powershell
Copy-Item .env.example .env
notepad .env
```

After saving `.env`, restart the backend:

```powershell
python -m uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
```

The **Title Promise → Search YouTube inside Veri-Click** section shows whether search is configured. Enter at least two characters and press **Search videos**. Pick a result to load it into the embedded player, then check its title promise.

## Limits and troubleshooting

- This is search, not a complete dump of all YouTube videos. Results are query-specific, paginated, and at most 50 results can be requested per API call.
- YouTube API quotas apply. Check the project's Google Cloud quota dashboard; `search.list` has a separate daily quota bucket, so avoid repeatedly clicking Search. See [Search: list](https://developers.google.com/youtube/v3/docs/search/list) and the [quota calculator](https://developers.google.com/youtube/v3/determine_quota_cost).
- If the app says the key is invalid or the API is disabled, recheck the selected Google Cloud project, enabled API and `.env` spelling, then restart the server.
- If quota is exhausted, wait until reset or check Google Cloud quota settings.
- A search result appearing does not guarantee it can be embedded or that captions are available. The query filters for embeddable videos, but playback and captions can still depend on the owner's settings and location.
- Public captions may be unavailable. In that case, paste a timestamped transcript in the Title Promise section, or use the separate Worth Watching media-upload workflow to transcribe media you have permission to process.
- The key is loaded by the local server and is never returned in API responses or rendered in the browser. `.env` is ignored by Git.
