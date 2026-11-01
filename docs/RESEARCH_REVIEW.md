# Research Review — Veri-Click: Multimodal Clickbait Detection

This review is based first on the three supplied files: **Multimodal Clickbait Detection in YouTube Videos using RoBERTa and Stacked Ensemble Learning** (research paper, 31 PDF pages including front matter), **Final 0.4** (11-page presentation), and **Veri-Click: Truth in Headlines** (11-slide project deck). External sources are identified separately.

## 1. What the supplied work did

### Problem and intended product
The research paper frames clickbait as a mismatch between an attention-grabbing title/thumbnail and what a viewer reasonably expects from the video. It proposes binary classification (`1 = clickbait`, `0 = non-clickbait`) on YouTube. The slide deck broadens the product goal to a **Chrome extension** that overlays screening badges on YouTube cards, and the `Veri-Click: Truth in Headlines` deck highlights informal language, multilingual coverage, a summary dashboard, and analysis of title–thumbnail mismatch.

### Dataset as reported
The paper reports an enhanced dataset of **37,397 labeled video records**. The described fields are the video title, binary label, and thumbnail image path, with metadata mentioned in the abstract/methodology. Example rows include “10 Riddles That Will Trick Your Brain” (label 1), “Top 10 Worst Moments in DC Comics” (label 1), “Binging with Babish: Lasagna from Garfield” (label 0), and “Hayley Kiyoko - Dear Hayley Kiyoko” (label 0). The paper says the dataset is mostly English-language and YouTube-specific.

**Reproducibility caution:** the paper references `alessiovierti/youtube-clickbait-detector` as the public dataset origin. The linked repository README describes a **title + engagement-metadata SVM pipeline** using Word2Vec (25 dimensions), min–max scaling, and an archived classifier; it reports more than 28,000 training rows and a 7,200-row test set. Its README does not describe the exact 37,397-row thumbnail-enhanced dataset used in the paper. Therefore, the enhanced images/labels and the paper's full training table cannot be reconstructed from the PDFs alone.

### Preprocessing and feature extraction
The proposed research pipeline has the following stages:

1. **Text preprocessing:** lower-case titles, keep stopwords to preserve context, tokenize with the RoBERTa tokenizer, and pad/truncate to a fixed sequence length.
2. **Image preprocessing:** resize thumbnails to 224 × 224, normalize with ImageNet statistics, and convert to tensors.
3. **Text representation:** use the RoBERTa contextual representation (the paper says the classification-token representation) to capture semantic dependencies, exaggeration and curiosity-driven phrasing.
4. **Visual representation:** remove the final classification layer of pretrained ResNet50 and use the penultimate-layer representation to represent high-level thumbnail patterns.
5. **Text–image similarity:** compute cosine similarity to expose possible title/thumbnail mismatch.
6. **Sentiment:** extract positive, negative, neutral and compound VADER scores.
7. **Lexical/pattern features:** numerical values, question marks, exclamation marks, uppercase-character ratio, and clickbait trigger-word frequency such as “shocking”, “secret” and “OMG”.
8. **Feature fusion:** concatenate the text, image and engineered features.
9. **Scaling and dimensionality reduction:** standardize features and reduce the fused vector to 256 PCA components.
10. **Classification:** compare Logistic Regression, Naïve Bayes, K-Nearest Neighbours, Random Forest, XGBoost and RBF-SVM. Combine base learners with a Logistic Regression meta-learner for the stacked ensemble.

The methodology diagram on page 5 of the paper / slide 6 of the presentation visually lays out the parallel text and image pipelines, handcrafted features, fusion, PCA and stacked classifier.

### Evaluation reported by the paper
The paper's Table 5.1 reports:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Stacked ensemble | 0.9734 | 0.9753 | 0.9703 | 0.9728 | 0.99 |
| SVM (RBF) | 0.9725 | 0.9766 | 0.9670 | 0.9718 | 0.98 |
| XGBoost | 0.9615 | 0.9684 | 0.9525 | 0.9604 | 0.97 |
| Logistic Regression | 0.9553 | 0.9564 | 0.9522 | 0.9543 | 0.96 |
| KNN | 0.9451 | 0.9545 | 0.9323 | 0.9433 | 0.96 |
| Random Forest | 0.9180 | 0.9721 | 0.8573 | 0.9111 | 0.95 |
| Naïve Bayes | 0.6591 | 0.6333 | 0.7222 | 0.6748 | 0.71 |

The presentation reproduces the same general ordering and rounds the stacked F1 score to about 0.973. There is a small wording inconsistency: one paragraph in the paper calls SVM accuracy 97.24%, while its main table gives 0.9725 (97.25%). The table values above are retained as the primary reported results.

## 2. What is missing or underspecified

These are reproducibility and engineering observations, not accusations about the authors' work:

- **Exact multimodal artifact:** the PDF does not bundle the enhanced 37,397-row CSV, matching thumbnails, train/test IDs, or trained checkpoints. The public repo cited in the references documents a different, metadata-driven SVM pipeline.
- **Cross-modal dimensionality:** RoBERTa's pooled representation and ResNet50's penultimate representation naturally have different dimensions. A cosine similarity between them requires a documented shared projection or compatible embedding space. The paper does not fully spell out that projection in the described method.
- **Leakage-safe preprocessing:** scaler/PCA should be fitted on training data only and applied to validation/test data. The report does not provide enough artifact-level detail to independently verify every fit boundary.
- **Stacking protocol:** the description says validation predictions feed the meta-learner, but does not specify out-of-fold generation or a final untouched test protocol in enough detail to reproduce the reported number independently.
- **Split and data distribution:** the diagram shows a stratified 80:20 split, but the documents do not fully publish class counts, deduplication rules, random seed, full hyperparameters, or a per-class confusion matrix.
- **Semantic mismatch:** a thumbnail can be highly colourful without being deceptive, and a title can be clickbait even when semantically related to an image. Similarity alone is an indicator, not proof.
- **Binary label ambiguity:** dramatic-but-accurate titles and misleading titles are not necessarily the same category. The report notes that labeling can be subjective.
- **Product claims vs. implementation:** the 11-slide `Veri-Click: Truth in Headlines` deck proposes a Chrome extension, multilingual screening, thumbnail badges, and a dashboard. The supplied deck itself is a proposal/presentation; it does not include extension source, a deployed endpoint, or evidence of live YouTube integration.

## 3. Practical additions implemented in this ZIP

1. **Working local API:** FastAPI endpoints for health/status, one-video analysis, batch analysis, dashboard stats, local history and model-artifact state.
2. **Polished browser dashboard:** single-video analysis with upload/preview, human-readable reasons, signal breakdown, recent history, bulk CSV audit/export, model architecture view and extension setup instructions.
3. **Local Chrome extension source:** Manifest V3 content script that adds user-triggered risk badges to up to 12 visible YouTube cards. It talks to the local API and validates thumbnail hosts server-side.
4. **Explainable no-download demo mode:** a transparent rule-based risk index and image-style statistics so the UI is usable without heavyweight pretrained checkpoints. The UI explicitly labels this mode as a demo, not a research benchmark or calibrated probability.
5. **Trainable deep-learning model:** BiGRU text branch + CNN thumbnail branch + engineered-feature MLP, joined with a learned gated-fusion layer and binary classification head. It is a smaller local alternative to the paper's RoBERTa + ResNet50 stack, designed for constrained CPU environments.
6. **Reproducible training/evaluation pipeline:** train/validation/test split, training-only vocabulary, weighted binary loss, gradient clipping, early stopping on validation F1, best-checkpoint selection, and held-out accuracy/precision/recall/F1/ROC-AUC/confusion matrix saved to JSON.
7. **Automated tests:** feature extraction, API health and errors, valid/invalid image handling, history persistence, batch results, model forward shapes and checkpoint-compatible features.

## 4. What was deliberately not claimed

- No claim that this ZIP reproduces 97.34% accuracy or F1 0.9728.
- No trained classifier checkpoint is bundled because no verifiable, labeled title–thumbnail dataset was supplied. The included demo scorer is clearly marked; a trained checkpoint becomes active only after running the training script on a properly labeled dataset with local thumbnail files.
- No LLM API call is needed. The core project is a real deep-learning training path, not a chatbot wrapper.
- The Chrome extension does not determine factual truth. It displays a risk-screening signal for titles/thumbnails.

## 5. Research extension plan

### Highest-value next experiments

**A. Reproduce the paper baseline.** Recover the exact image-enhanced dataset, publish a fixed train/validation/test manifest, implement RoBERTa pooled title embeddings and ResNet50 image embeddings, train scalers/PCA only on train rows, then run each listed baseline and a leakage-safe stacked ensemble.

**B. Ablation study.** Compare text-only, image-only, text + image, text + image + engineered signals, and gated fusion. Report macro-F1, per-class precision/recall, ROC-AUC, calibration, latency, peak memory and bootstrap confidence intervals.

**C. More faithful cross-modal alignment.** Add a pretrained joint vision–language encoder such as CLIP to obtain image and text vectors in a common embedding space. Compare its image–text similarity against separate RoBERTa/ResNet features. CLIP's model documentation describes a joint image/text embedding space and similarity score, which is suitable for alignment experiments.

**D. Better labels.** Separate (i) informative / non-clickbait, (ii) sensational but accurate, and (iii) misleading / bait-and-switch. Human-review uncertain cases and publish inter-annotator agreement. Avoid auto-labeling every trigger word as ground truth.

**E. Product robustness.** Add multilingual data, title/thumbnail OCR, transcript-vs-title claims, sampled video frames, drift checks, calibration/abstention, a feedback review queue and cross-channel/time-based split tests. Audio/transcripts should be opt-in and comply with platform policies and content rights.

## 6. External research context checked for this build

- Samuel and Saputri's 2025 *Detecting YouTube Clickbait with Transformer Models: A Comparative Study* compares BERT, RoBERTa and XLNet on two datasets and discusses accuracy and efficiency. It reports above-95% accuracy but different top models across datasets; this reinforces that performance is dataset-dependent and that scores should not be compared without matching the setup: https://doi.org/10.22146/ijccs.111977
- The cited Vierti GitHub repository's README identifies its method as Word2Vec title embeddings + engagement metadata + an SVM, and reports its own test results: https://github.com/alessiovierti/youtube-clickbait-detector
- Hugging Face's CLIP documentation explains that CLIP maps image and text into a shared space for similarity and zero-shot tasks: https://huggingface.co/docs/transformers/model_doc/clip
- IEEE's 2025 conference listing (added to IEEE Xplore in May 2026) describes a title/thumbnail/audio clickbait study using BanglaBERT and ViT; it is a promising direction for multilingual and audio-aware extensions: https://ieeexplore.ieee.org/abstract/document/11491659

## 7. Source page locator

- **Research paper:** abstract and claimed contribution, pp. 1–2; related work, pp. 5–8; methodology and architecture diagram, pp. 9–12; dataset and models, pp. 13–16; evaluation/results table, pp. 17–20; limitations and future work, pp. 21–22; references, pp. 23–24 of the printed pagination (31 PDF pages including front matter).
- **Final 0.4 presentation:** architecture diagram on slide/page 5; dataset and results on pages 6–8; conclusion and future work on page 10; references on page 11.
- **Veri-Click: Truth in Headlines deck:** problem/real-time Chrome extension idea on slides 2–3; research gaps on slide 4; platform/dashboard recommendations on slide 5; proposed method on slide 6; reported results on slide 7; multilingual, audio/video and XAI extensions on slide 9.
