# Demo Plan and Viva Guide

## Five-minute live demonstration

### 0:00–0:35 — State the problem

“Veri-Click AI helps viewers inspect clickbait-style presentation and compare a title promise with the content available in a transcript. It is a screening aid, not a factual truth verdict.” Point to the research review: the supplied paper's RoBERTa + ResNet50 + engineered features + PCA + stacked-ensemble result is not claimed as a reproduced result here.

### 0:35–1:15 — Analyze one title

1. Open **Video analyzer**.
2. Try a sensational title and a neutral tutorial title.
3. Explain the risk index and individual title signals. State clearly if the page is in `DEMO MODE` because no trained checkpoint is present.

### 1:15–2:05 — Show YouTube Title Promise

1. Open **Title Promise** and paste a public YouTube URL.
2. Click **Load player** to display it inside the app.
3. Click **Check title promise** to try public captions, or paste a transcript with timestamps if captions are inaccessible.
4. Show the extracted promise, one of `Fulfilled` / `Partly Fulfilled` / `Not Enough Evidence`, its reasoning, and evidence timestamp cards. Click a timestamp to jump the embedded player.

Explain that this status is based on transcript evidence and optional similarity ranking; it does not independently fact-check truth. Public captions may be unavailable.

### 2:05–2:45 — Worth watching + Batch audit

Briefly show **Worth watching**: local media is transcribed with Whisper, summarized, and compared with a personal goal. Its goal-fit score is a keyword/content-specificity heuristic. Then briefly show **Batch audit** to demonstrate CSV-based title screening and export.

### 2:45–4:15 — Deep-learning model comparison

Show the Model Lab architecture and the four-model comparison table after running `scripts/compare_models.py` on real labeled data: BiGRU text-only, CNN thumbnail-only, engineered-feature MLP and multimodal gated fusion. Explain accuracy, precision, recall, F1, ROC-AUC, confusion matrix, same held-out split, and training time. The actual table is populated only after real training; do not invent numbers if the run is not complete.

### 4:15–5:00 — Tests and research integrity

Show `docs/TEST_REPORT.md` and the command `python -m pytest -q`. Mention that software tests and synthetic smoke tests prove the pipeline executes, not that a model generalizes. If available, briefly show the Chrome extension overlay on YouTube. It is a local, user-triggered extension and is not published to the Chrome Web Store.

## Key viva questions and answers

**1. What is clickbait?**
Content framed to provoke clicks through exaggerated/sensational wording, curiosity gaps, or misleading title/thumbnail presentation. This project's screening label should not be interpreted as proof of falsity.

**2. Why is clickbait detection multimodal?**
A title may be ambiguous by itself, while a thumbnail adds visual context. Joint signals can reveal patterns not visible from text alone, though they still cannot verify a claim against the actual video without video/transcript evidence.

**3. What does the Title Promise Checker evaluate?**
It extracts a checkable expectation from the title, scans a timestamped transcript for relevant evidence and possible filler/promos, and reports `Fulfilled`, `Partly Fulfilled`, or `Not Enough Evidence`. It quotes timestamped evidence when available. The verdict is provisional transcript matching—not factuality or intent detection. Sentence-transformer embeddings, if installed, help rank related windows but do not prove entailment.

**4. Why use a BiGRU?**
A bidirectional GRU processes title token sequences in both directions and models sequential context at a lower compute cost than a large transformer. This implementation is trainable locally and compact, while RoBERTa is a future reproduction target.

**5. Why use a CNN?**
Convolutional filters learn local spatial patterns in thumbnail images; successive convolution/pooling layers build more abstract representations. Adaptive pooling reduces the final spatial map to a compact feature vector.

**6. What is gated fusion?**
The network learns a sigmoid gate `g` from the text and image representations. For each feature dimension, fused representation is `g ⊙ text + (1 − g) ⊙ image`. This allows the network to weight modalities differently, unlike simply averaging them.

**7. What are the 16 side features?**
Lexical ratios/counts for trigger phrases, curiosity and urgency markers, uppercase/punctuation/numbers/emoji, plus image style measurements such as saturation, contrast, brightness, colorfulness and edge density. They are appended as side input and do not replace learned text/image features.

**8. How do you prevent test leakage?**
The trainer creates a stratified training split, then validation/test holdouts; the vocabulary is built from the training titles only; validation F1 selects the checkpoint; final metrics are calculated on the held-out test split. For stronger claims, run temporal and channel-disjoint evaluation as recommended in the roadmap.

**9. Why report precision, recall, F1 and ROC-AUC instead of accuracy alone?**
Accuracy hides the class-specific error balance when labels are imbalanced. Precision shows how many flagged examples are correct, recall how many clickbait examples are found, F1 balances precision and recall, and ROC-AUC assesses ranking across thresholds when both classes exist.

**10. Why is the model not already 97.34% accurate?**
The original paper reports that result for its own RoBERTa + ResNet50 + handcrafted/PCA + stacked ensemble experiment. The PDFs do not include the full matching dataset, split manifest or checkpoint. The deliverable intentionally does not claim that result. The compact network has to be trained and independently evaluated on supplied data.

**11. What is a primary current limitation?**
The fallback scorer is heuristic. Trained mode is binary and dataset-dependent. Thumbnail statistics are not semantic image understanding, multilingual behavior is unvalidated, and full video/audio context is not yet analyzed.

**12. What is the next high-value research improvement?**
Add an ablation suite (text-only, image-only, both, both + side signals), a joint vision-language alignment model such as CLIP, calibration/abstention, channel/time-held-out testing, and a human-reviewed three-class label taxonomy.

**13. What is a possible ethical risk?**
A model can flag accurate but dramatic content, or miss nuanced deception. Dataset biases, language differences and changing clickbait patterns can create unfair false positives/negatives. The UI therefore calls its output a screening aid, explains signals, and does not claim truth verification.

## Submission checklist

- [ ] Start from a clean virtual environment using Python 3.11/3.12.
- [ ] Run the test commands in `README.md` and keep the latest test report.
- [ ] Demonstrate `DEMO MODE` honestly if no trained checkpoint exists.
- [ ] For trained-mode results, include dataset provenance, labels, class counts, split seed, held-out metrics and limitations in the report.
- [ ] Do not use synthetic UI/test images as model-training evidence.
- [ ] Load the extension on YouTube only after the API is running.
- [ ] Take screenshots of the dashboard, title-promise evidence report, batch results, four-model comparison table (after real training), and extension overlay for the submission report.


## Deep-learning comparison demo

After running `scripts/compare_models.py` on real labeled data, open Model Lab and show the four-model held-out comparison: BiGRU text-only, CNN image-only, engineered-feature MLP, and multimodal gated fusion. Explain that all models use the same split, the model epoch is chosen on validation F1, and final reported metrics come from the held-out test partition. Show the confusion matrix as TN / FP · FN / TP. Do not present a small pilot score as a paper replication or claim that the model fact-checks the video's claims.
