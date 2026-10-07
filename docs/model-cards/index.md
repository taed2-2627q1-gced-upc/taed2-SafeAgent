# Model cards

These cards describe the SafeAgent classifiers. The classical baseline has a
fitted model and validation results. The transformer cards are still drafts.

- [TF-IDF character n-grams + Linear SVM](model_card_tfidf_linear_svm.md):
  command-only classical baseline, with validation results.
- [CodeBERT-base](model_card_codebert_base.md): command-only encoder model.
- [ModernBERT-base](model_card_modernbert_base.md): command and session-context
  encoder model.

Each card covers intended use, risks, training plans, and evaluation criteria.
See the [Dataset Card](../dataset-card.md) for the shared data protocol.
