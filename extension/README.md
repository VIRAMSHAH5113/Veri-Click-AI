# Veri-Click AI — YouTube Chrome Companion

This is a local developer extension, not a Chrome Web Store release.

1. Start the Veri-Click API at `http://127.0.0.1:8000`.
2. In Chrome, open `chrome://extensions` and enable **Developer mode**.
3. Choose **Load unpacked** and select this `extension` directory.
4. Open YouTube and click **Scan visible videos**. The extension checks up to 12 visible video cards per scan and adds a risk badge.

The extension sends title + the thumbnail URL to the local API. The API only accepts HTTPS thumbnail URLs on the `ytimg.com`, `youtube.com`, or `googleusercontent.com` host families. Scan requests do not write items to saved history. The badge is a screening aid, not a fact-check verdict.
