# Upgrade your existing Veri-Click folder (Windows)

These steps preserve your current `.venv` and local SQLite history.

1. Stop the currently running Veri-Click server in its PowerShell window with **Ctrl+C**.
2. Download `Veri-Click-AI-Worth-Watching.zip` and extract it into your existing parent folder:
   `C:\Users\DARSHIT\OneDrive\Desktop\click_ai`
3. When Windows asks whether to replace files in `Veri-Click-AI`, choose **Replace the files in the destination**. Do not delete the project folder or `.venv`.
4. Open PowerShell in `C:\Users\DARSHIT\OneDrive\Desktop\click_ai\Veri-Click-AI` and activate the current environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. Install the new upload dependency and optional speech AI:

   ```powershell
   pip install -r requirements.txt
   pip install -r requirements-media.txt
   ```

6. Start the server again:

   ```powershell
   python -m uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
   ```

7. Refresh `http://127.0.0.1:8000` in your browser and open **Worth watching** in the sidebar.
8. Upload a short video/audio clip (start with 1–5 minutes) and describe your goal, e.g. `I want to learn Python basics for my exam`.

The first analysis downloads the Whisper `tiny` speech model and needs an internet connection. Later transcription runs locally. On a low-RAM laptop use a short clip first. If anything fails, the message in the page or the last lines of PowerShell will show the blocker.
