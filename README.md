# Text Classification Experiment Lab

A practical experimentation workspace for training, evaluating, comparing, and using transformer-based text classification models. The project is designed to make it easy to run multiple training experiments with different hyperparameters, compare validation metrics, apply text data augmentation strategies, and reuse trained checkpoints for inference.

## What this repository does

This repository provides a modular workflow for text classification experiments using Hugging Face Transformers and PyTorch. It includes utilities to prepare labeled text datasets, split train/validation data, fine-tune transformer models, track metrics across epochs, save the best checkpoint, apply data augmentation, and run inference with already trained models.

The main use case is rapid experimentation: change model architecture, batch size, learning rate, dropout, weight decay, freezing strategy, early stopping, class weights, and other training settings from a single configuration object. Each experiment can save its own checkpoint and training history, making it easier to compare results across runs.

## Main components

```text
.
├── Multi_Experiments_text_classification.ipynb  # Main notebook for training experiments
├── data_augment.ipynb                          # Notebook for generating augmented datasets
├── classify_data.ipynb                         # Notebook for inference and model comparison
├── text_classification.py                      # Training, evaluation, metrics, and checkpoint logic
├── text_data_augment.py                        # Text augmentation utilities
├── inference_funcs.py                          # Inference and post-training evaluation utilities
├── translate_df_col.py                         # Translation helper used for back-translation
└── README.md
```


## Features

- Fine-tune Hugging Face transformer models for multi-class text classification.
- Configure experiments through a single `TextClassifierTrainConfig` dataclass.
- Compare experiments using validation loss, accuracy, macro F1, weighted F1, and metrics excluding an “Other” class.
- Save the best model checkpoint, tokenizer, Hugging Face config, and training history.
- Use class weights to handle class imbalance.
- Apply layer-freezing strategies such as freezing embeddings, freezing bottom layers, or training only the top layers.
- Support early stopping based on a configurable validation metric.
- Generate augmented datasets using:
  - Back-translation
  - Synonym replacement
  - Contextual substitution
  - Contextual insertion
- Run inference with trained checkpoints and evaluate confidence thresholds.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install pandas numpy torch transformers scikit-learn matplotlib tqdm openpyxl deep-translator nltk nlpaug
```

If you use synonym replacement, download WordNet resources:

```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Basic workflow

### 1. Prepare and split data

Use `data_augment.ipynb` to load your labeled dataset, translate the text column if needed, clean the data, create `label_idx`, and generate train/validation splits.

The expected dataset should contain at least:

```text
text column   -> for example: comentario_en
label column  -> for example: categoria
```

### 2. Generate augmented datasets

Use `data_augment.ipynb` and `text_data_augment.py` to create additional training examples. You can generate different datasets by class, depending on class imbalance and the amount of variation you want.

Available augmentation strategies:

```python
augment_df_with_backtranslation(...)
augment_df_with_synonym_replacement(...)
augment_df_with_contextual_augmentation(...)
```

### 3. Train multiple experiments

Use `train_experiments.ipynb` to define a new experiment:

```python
EXPERIMENT_NAME = "modernbert_large_exp_01"

config = TextClassifierTrainConfig(
    model_name="answerdotai/ModernBERT-large",
    text_col="comentario_en",
    label_col="categoria",
    valid_labels=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    other_class_code=9,
    max_length=128,
    train_batch_size=16,
    val_batch_size=32,
    learning_rate=1e-5,
    weight_decay=0.2,
    num_epochs=100,
    use_class_weights=True,
    model_save_name=f"{EXPERIMENT_NAME}.pt",
    freeze_strategy="all_but_top_n",
    unfreeze_top_n=6,
    use_early_stopping=True,
    early_stopping_patience=15,
)
```

Then train the model:

```python
model, tokenizer, history_df = train_model(train_df, val_df, config)
```

### 4. Compare training results

Each training run returns a `history_df` and saves a `history.csv` file with metrics such as:

- `train_loss`
- `val_loss`
- `val_acc_all`
- `val_f1_macro_all`
- `val_f1_weighted_all`
- `val_f1_macro_excl_other`
- `val_f1_weighted_excl_other`

You can visualize the evolution of loss and F1 scores with:

```python
plot_history(history_df)
```

### 5. Run inference with trained models

Use `classify_data.ipynb` and `inference_funcs.py` to classify a new dataframe:

```python
df_classified = classify_dataframe_with_model(
    df_to_classify=df_to_classify,
    checkpoint_path="models/best_model.pt",
    text_col="comentario_en",
    valid_labels=VALID_LABELS,
    class_names=CLASS_NAMES,
    batch_size=32,
    fallback_model_name="answerdotai/ModernBERT-base",
)
```

The output dataframe includes:

```text
categoria_pred
nombre_categoria
certeza_modelo
```

You can also compute metrics when ground-truth labels are available:

```python
metrics, report, cm_df = compute_metrics_from_classified_df(
    classified_df=df_classified,
    true_label_col="categoria",
    pred_label_col="categoria_pred",
    valid_labels=VALID_LABELS,
    class_names=CLASS_NAMES,
    other_class_code=9,
)
```

## Suggested experiment naming convention

Use names that make comparison easier:

```text
modernbert_base_lr2e5_wd001_len128
modernbert_large_lr1e5_wd02_freeze_top6_aug
modernbert_base_aug_bt_sr_ctx_threshold_eval
```

A good convention is:

```text
{model}_{lr}_{weight_decay}_{max_length}_{augmentation}_{freeze_strategy}
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
