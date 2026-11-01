# Feature Roadmap Beyond the Baseline

## Included now

- **PromiseProof claim-wise title audit:** split common title formats into checkable topic, list-count, free/no-cost, quantified-outcome, duration, and no-code claims; map matching transcript windows to timestamps; report supported/partly supported/no-evidence counts and claim evidence coverage; show limited-transcript warnings and a separate title-only clickbait-risk result. It is an evidence-navigation aid, not factual verification or a generative LLM.

- One-video title + optional thumbnail analysis.
- Visible thumbnail preview and file validation.
- Explainable feature signals and image-style diagnostics.
- Local history and dashboard statistics.
- CSV title audit for up to 200 rows with downloadable results.
- Compact CNN + BiGRU gated fusion training/inference path.
- Holdout evaluation metrics and saved model artifact state.
- Chrome extension with on-demand badges for up to 12 visible video cards.

## Best upgrades for a final-year research submission

1. **Exact paper reproduction:** obtain the original enhanced labeled dataset; recreate RoBERTa + ResNet50 features, StandardScaler, PCA(256), six baselines and LR meta-learner. Fix split manifests and make sure all fitting happens on training rows only.
2. **CLIP alignment experiment:** use a shared vision-language embedding space to compare title-image similarity versus the original separate encoders. Report cosine distributions for matched, mismatched and hard-negative pairs.
3. **Multiclass label framework:** distinguish informative, sensational-but-accurate, and misleading/bait-and-switch cases. Add annotator guidelines and inter-rater agreement.
4. **Calibration and abstention:** calibrate predicted probabilities on validation data; display “uncertain—review” when uncertainty is high. In demo mode, the displayed number remains a rule-based index, not a probability.
5. **Explainable AI:** show which text spans and modalities affected the trained score; consider Integrated Gradients for the neural model and SHAP on engineered-feature baselines.
6. **OCR on thumbnail text:** detect embedded thumbnail claims and compare them with titles; make OCR confidence visible.
7. **Multimodal claim evidence:** extend the current PromiseProof transcript audit with opt-in frame-level visual evidence and/or OCR for on-screen text where lawful and available. Avoid presenting auto-generated transcript summaries as verified truth; show every modality and evidence gap explicitly.
8. **Multilingual support:** add non-English titles and multilingual encoders; evaluate by language and avoid assuming English token cues transfer.
9. **Temporal and channel-held-out validation:** test on newer upload dates and unseen channels to reveal leakage and content drift.
10. **Research-grade reporting:** include class distribution, confusion matrix, per-class and macro metrics, confidence intervals, inference latency, memory, ablations, error examples and a reproducibility checklist.
