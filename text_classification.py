import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
import os, random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt



# Clase para configuración de experimentos de entrenamiento
@dataclass
class TextClassifierTrainConfig:
    model_name: str = "tasksource/ModernBERT-base-nli"
    text_col: str = "contenido_en"
    label_col: str = "codigo"

    valid_labels: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    other_class_code: int = 8
    class_names: dict[int, str] = field(default_factory=lambda: {
        1: "Tiempo de espera",
        2: "Amabilidad",
        3: "Atención clínica",
        4: "Información recibida",
        5: "Servicio de Transporte",
        6: "Infraestructura",
        7: "Calificación",
        8: "Otros",
    })

    max_length: int = 256
    val_size: float = 0.20
    random_state: int = 42

    train_batch_size: int = 16
    val_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.10
    grad_clip: float = 1.0

    use_class_weights: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    save_dir: str = "artifacts_modernbert_cls"
    best_metric_name: str = "val_f1_macro_excl_other"
    model_save_name: str = "best_model.pt"

    freeze_strategy: str = "none" # | embeddings | bottom_n | all_but_top_n
    freeze_n_layers: int = 0
    unfreeze_top_n: int = 0
    freeze_embeddings: bool = True

    use_early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0

    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    classifier_dropout: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    

def build_label_maps(valid_labels, class_names:dict):
    """
    Mapea códigos reales 1..8 a índices 0..7 para entrenamiento.
    Ejemplo: class_names = {1: "Tiempo de espera",
                            2: "Amabilidad",
                            3: "Atención clínica",
                            4: "Información recibida",
                            5: "Servicio de Transporte",
                            6: "Infraestructura",
                            7: "Calificación",
                            8: "Otros",
                        }
    """
    code2idx = {code: idx for idx, code in enumerate(valid_labels)}
    idx2code = {idx: code for code, idx in code2idx.items()}

    id2label = {idx: class_names[code] for idx, code in idx2code.items()}
    label2id = {label: idx for idx, label in id2label.items()}

    return code2idx, idx2code, id2label, label2id

# Preparar y limpiar dataframes

def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x else None


def prepare_dataframe(df, config: TextClassifierTrainConfig):
    """
    - deja solo texto + label
    - filtra labels inválidos (por defecto elimina 997/998)
    - elimina nulos, textos vacíos y duplicados en el subset dado
    - crea columna label_idx para entrenamiento
    """
    df = df.copy()
    subset=config.text_col

    df[config.text_col] = df[config.text_col].map(clean_text)
    df = df.dropna(subset=[config.text_col, config.label_col])

    # Asegurar entero en labels
    df[config.label_col] = pd.to_numeric(df[config.label_col], errors="coerce")
    df = df.dropna(subset=[config.label_col])
    df[config.label_col] = df[config.label_col].astype(int)

    # Drop duplicados
    df.drop_duplicates(subset=subset, inplace=True)

    # Filtrar solo 1..8
    df = df[df[config.label_col].isin(config.valid_labels)].copy()

    code2idx, idx2code, id2label, label2id = build_label_maps(config.valid_labels, config.class_names)
    df["label_idx"] = df[config.label_col].map(code2idx).astype(int)

    return df, code2idx, idx2code, id2label, label2id


# Split entrenamiento y validación 
# Para consistencia de experimentos utilizar misma segmentación, es decir, generar train_df y val_df, y luego guardar
def stratified_split(df, config: TextClassifierTrainConfig):
    train_df, val_df = train_test_split(
        df,
        test_size=config.val_size,
        random_state=config.random_state,
        stratify=df["label_idx"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# Pytorch Datasets y Dataloaders
class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="contenido_eng", label_col="label_idx", max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, self.text_col]
        label = int(self.df.loc[idx, self.label_col])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
    
def build_dataloaders(train_df, val_df, tokenizer, config: TextClassifierTrainConfig):
    train_ds = TextClassificationDataset(
        train_df,
        tokenizer,
        text_col=config.text_col,
        label_col="label_idx",
        max_length=config.max_length,
    )
    val_ds = TextClassificationDataset(
        val_df,
        tokenizer,
        text_col=config.text_col,
        label_col="label_idx",
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if config.device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if config.device == "cuda" else False,
    )

    return train_loader, val_loader


# Creación del modelo base
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad

def get_backbone_module(model: nn.Module) -> nn.Module:
    # Caso estándar HF
    if hasattr(model, "base_model"):
        return model.base_model

    base_model_prefix = getattr(model, "base_model_prefix", None)
    if base_model_prefix and hasattr(model, base_model_prefix):
        return getattr(model, base_model_prefix)

    # Fallbacks comunes
    for attr in ["modernbert", "bert", "roberta", "deberta", "distilbert", "electra", "model"]:
        if hasattr(model, attr):
            return getattr(model, attr)

    raise ValueError("No se pudo identificar el backbone del modelo.")

def get_embedding_module(backbone: nn.Module):
    """
    Intenta ubicar el bloque de embeddings.
    """
    for attr in ["embeddings"]:
        if hasattr(backbone, attr):
            return getattr(backbone, attr)
    return None

def get_encoder_layers(backbone: nn.Module):
    """
    Devuelve una lista ordenada de capas del encoder.
    """
    # Patrones comunes en HF
    candidates = [
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("transformer", "layer"),
        ("transformer", "layers"),
    ]

    for parent_attr, child_attr in candidates:
        if hasattr(backbone, parent_attr):
            parent = getattr(backbone, parent_attr)
            if hasattr(parent, child_attr):
                layers = getattr(parent, child_attr)
                if isinstance(layers, (nn.ModuleList, list, tuple)):
                    return list(layers)

    # Otros patrones comunes
    for attr in ["layers", "h", "layer"]:
        if hasattr(backbone, attr):
            layers = getattr(backbone, attr)
            if isinstance(layers, (nn.ModuleList, list, tuple)):
                return list(layers)

    raise ValueError("No pude identificar las capas del encoder del backbone.")

def apply_layer_freezing(model: nn.Module, config):
    """
    config.freeze_strategy soportadas:
    - none
    - embeddings
    - bottom_n
    - all_but_top_n
    """
    backbone = get_backbone_module(model)
    embeddings = get_embedding_module(backbone)
    layers = get_encoder_layers(backbone)

    n_total_layers = len(layers)

    # Siempre partimos dejando todo entrenable
    set_requires_grad(model, True)

    if config.freeze_strategy == "none":
        pass

    elif config.freeze_strategy == "embeddings":
        if embeddings is not None and config.freeze_embeddings:
            set_requires_grad(embeddings, False)

    elif config.freeze_strategy == "bottom_n":
        n = int(config.freeze_n_layers)
        n = max(0, min(n, n_total_layers))

        if embeddings is not None and config.freeze_embeddings:
            set_requires_grad(embeddings, False)

        for layer in layers[:n]:
            set_requires_grad(layer, False)

    elif config.freeze_strategy == "all_but_top_n":
        k = int(config.unfreeze_top_n)
        k = max(0, min(k, n_total_layers))

        if embeddings is not None and config.freeze_embeddings:
            set_requires_grad(embeddings, False)

        n_to_freeze = n_total_layers - k
        for layer in layers[:n_to_freeze]:
            set_requires_grad(layer, False)

    else:
        raise ValueError(
            f"freeze_strategy='{config.freeze_strategy}' no soportada."
        )

    # La cabeza de clasificación se deja entrenable
    # salvo que explícitamente quieras otra cosa
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Freeze strategy: {config.freeze_strategy}")
    print(f"Encoder layers detectadas: {n_total_layers}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Debug opcional
    for name, param in model.named_parameters():
        if any(tag in name.lower() for tag in ["classifier", "score", "head"]):
            print(f"[HEAD] {name}: requires_grad={param.requires_grad}")

    return model

def build_model(config: TextClassifierTrainConfig, id2label, label2id):
    hf_config = AutoConfig.from_pretrained(config.model_name)

    hf_config.num_labels = len(config.valid_labels)
    hf_config.id2label = id2label
    hf_config.label2id = label2id

    hf_config.embedding_dropout = config.embedding_dropout
    hf_config.attention_dropout = config.attention_dropout
    hf_config.mlp_dropout = config.mlp_dropout
    hf_config.classifier_dropout = config.classifier_dropout
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        config=hf_config,
        ignore_mismatched_sizes=True,
    )

    model = apply_layer_freezing(model, config)
    return model

def build_class_weights(train_df, num_classes, device):
    y = train_df["label_idx"].values
    classes = np.arange(num_classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights


# Cálculo de métricas

def is_better_metric(current_score, best_score, min_delta=0.0):
    """
    True si current_score mejora suficientemente respecto a best_score.
    """
    return current_score > (best_score + min_delta)

def compute_metrics(y_true_idx, y_pred_idx, idx2code, valid_labels, other_class_code):
    y_true_code = np.array([idx2code[i] for i in y_true_idx])
    y_pred_code = np.array([idx2code[i] for i in y_pred_idx])

    acc_all = accuracy_score(y_true_code, y_pred_code)
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(
        y_true_code, y_pred_code, average="macro", zero_division=0
    )
    p_weighted_all, r_weighted_all, f1_weighted_all, _ = precision_recall_fscore_support(
        y_true_code, y_pred_code, average="weighted", zero_division=0
    )

    labels_wo_other = [x for x in valid_labels if x != other_class_code]

    mask = y_true_code != other_class_code
    y_true_ex = y_true_code[mask]
    y_pred_ex = y_pred_code[mask]

    if len(y_true_ex) > 0:
        acc_ex = accuracy_score(y_true_ex, y_pred_ex)
        p_macro_ex, r_macro_ex, f1_macro_ex, _ = precision_recall_fscore_support(
            y_true_ex, y_pred_ex,
            labels=labels_wo_other,
            average="macro",
            zero_division=0,
        )
        p_weighted_ex, r_weighted_ex, f1_weighted_ex, _ = precision_recall_fscore_support(
            y_true_ex, y_pred_ex,
            labels=labels_wo_other,
            average="weighted",
            zero_division=0,
        )
    else:
        acc_ex = p_macro_ex = r_macro_ex = f1_macro_ex = np.nan
        p_weighted_ex = r_weighted_ex = f1_weighted_ex = np.nan

    return {
        "accuracy_all": acc_all,
        "precision_macro_all": p_macro_all,
        "recall_macro_all": r_macro_all,
        "f1_macro_all": f1_macro_all,
        "precision_weighted_all": p_weighted_all,
        "recall_weighted_all": r_weighted_all,
        "f1_weighted_all": f1_weighted_all,
        "accuracy_excl_other": acc_ex,
        "precision_macro_excl_other": p_macro_ex,
        "recall_macro_excl_other": r_macro_ex,
        "f1_macro_excl_other": f1_macro_ex,
        "precision_weighted_excl_other": p_weighted_ex,
        "recall_weighted_excl_other": r_weighted_ex,
        "f1_weighted_excl_other": f1_weighted_ex,
    }


# Evaluación de Modelo

@torch.no_grad()
def evaluate_model(model, dataloader, device, idx2code, valid_labels, other_class_code , loss_fn=None):
    model.eval()

    all_losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits

        if loss_fn is None:
            loss = nn.CrossEntropyLoss()(logits, batch["labels"])
        else:
            loss = loss_fn(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)

        all_losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    metrics = compute_metrics(
        y_true_idx=all_labels,
        y_pred_idx=all_preds,
        idx2code=idx2code,
        valid_labels=valid_labels,
        other_class_code=other_class_code,
    )
    metrics["loss"] = float(np.mean(all_losses))
    return metrics, np.array(all_labels), np.array(all_preds)

# Entrenamiento de un epoch
def train_one_epoch(model, dataloader, optimizer, scheduler, device, idx2code,valid_labels, other_class_code, loss_fn=None):
    model.train()

    all_losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits

        if loss_fn is None:
            loss = nn.CrossEntropyLoss()(logits, batch["labels"])
        else:
            loss = loss_fn(logits, batch["labels"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)

        all_losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    metrics = compute_metrics(
        y_true_idx=all_labels,
        y_pred_idx=all_preds,
        idx2code=idx2code,
        valid_labels=valid_labels,
        other_class_code=other_class_code,
    )
    metrics["loss"] = float(np.mean(all_losses))
    return metrics


## Loop de entrenamiento:

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(train_df, val_df, config: TextClassifierTrainConfig):
    os.makedirs(config.save_dir, exist_ok=True)
    set_seed(config.random_state)

    code2idx, idx2code, id2label, label2id = build_label_maps(
                                                config.valid_labels, 
                                                config.class_names
                                            )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print("Modelo base:", config.model_name)
    model = build_model(config, id2label=id2label, label2id=label2id)
    print("num_labels:", model.config.num_labels)
    print("id2label:", model.config.id2label)
    model.to(config.device)

    train_loader, val_loader = build_dataloaders(
                                    train_df, 
                                    val_df, 
                                    tokenizer, 
                                    config
                                )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = None
    if config.use_class_weights:
        class_weights = build_class_weights(
            train_df,
            num_classes=len(config.valid_labels),
            device=config.device,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    best_score = -np.inf
    best_epoch = None
    epochs_without_improvement = 0

    best_model_path = os.path.join(config.save_dir, config.model_save_name)

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*80}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            idx2code=idx2code,
            valid_labels=config.valid_labels,
            other_class_code=config.other_class_code,
            loss_fn=loss_fn,
        )

        val_metrics, y_val_true, y_val_pred = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=config.device,
            idx2code=idx2code,
            valid_labels=config.valid_labels, 
            other_class_code=config.other_class_code,
            loss_fn=loss_fn,
        )

        row = {
            "epoch": epoch,

            "train_loss": train_metrics["loss"],
            "train_acc_all": train_metrics["accuracy_all"],
            "train_f1_macro_all": train_metrics["f1_macro_all"],
            "train_f1_weighted_all": train_metrics["f1_weighted_all"],
            "train_acc_excl_other": train_metrics["accuracy_excl_other"],
            "train_f1_macro_excl_other": train_metrics["f1_macro_excl_other"],
            "train_f1_weighted_excl_other": train_metrics["f1_weighted_excl_other"],

            "val_loss": val_metrics["loss"],
            "val_acc_all": val_metrics["accuracy_all"],
            "val_f1_macro_all": val_metrics["f1_macro_all"],
            "val_f1_weighted_all": val_metrics["f1_weighted_all"],
            "val_acc_excl_other": val_metrics["accuracy_excl_other"],
            "val_f1_macro_excl_other": val_metrics["f1_macro_excl_other"],
            "val_f1_weighted_excl_other": val_metrics["f1_weighted_excl_other"],
        }
        history.append(row)

        print(pd.Series(row))

        current_score = row[config.best_metric_name]

        if is_better_metric(
            current_score=current_score,
            best_score=best_score,
            min_delta=config.early_stopping_min_delta,
        ):
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "best_score": best_score,
                },
                best_model_path,
            )
            tokenizer.save_pretrained(os.path.join(config.save_dir, "tokenizer"))
            model.config.save_pretrained(os.path.join(config.save_dir, "hf_config"))
            print(f"✅ Nuevo mejor modelo guardado en: {best_model_path}")
        
        else:
            epochs_without_improvement += 1
            print(
                f"Sin mejora en {config.best_metric_name}. "
                f"Patience: {epochs_without_improvement}/{config.early_stopping_patience}"
            )

            if config.use_early_stopping and epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"Early stopping activado en epoch {epoch}. "
                    f"Mejor epoch: {best_epoch}, mejor {config.best_metric_name}: {best_score:.6f}"
                )
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config.save_dir, "history.csv"), index=False)

    print(f"\nMejor epoch: {best_epoch}")
    print(f"Mejor {config.best_metric_name}: {best_score:.6f}")

    best_checkpoint = torch.load(best_model_path, map_location=config.device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    return model, tokenizer, history_df

# Reportes de validación y gráficos
@torch.no_grad()
def predict_dataset(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def print_validation_reports(model, val_df, tokenizer, config: TextClassifierTrainConfig):
    _, idx2code, _, _ = build_label_maps(config.valid_labels, config.class_names)

    val_ds = TextClassificationDataset(
        val_df,
        tokenizer,
        text_col=config.text_col,
        label_col="label_idx",
        max_length=config.max_length,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if config.device == "cuda" else False,
    )

    y_true_idx, y_pred_idx = predict_dataset(model, val_loader, config.device)

    y_true_code = np.array([idx2code[i] for i in y_true_idx])
    y_pred_code = np.array([idx2code[i] for i in y_pred_idx])

    target_names_all = [config.class_names[c] for c in config.valid_labels]
    print("\n=== Classification report (todas las clases) ===")
    print(
        classification_report(
            y_true_code,
            y_pred_code,
            labels=list(config.valid_labels),
            target_names=target_names_all,
            digits=4,
            zero_division=0,
        )
    )

    labels_wo_other = [x for x in config.valid_labels if x != config.other_class_code]
    target_names_wo_other = [config.class_names[c] for c in labels_wo_other]
    mask = y_true_code != config.other_class_code

    print("\n=== Classification report (excluyendo 'Otros' como clase real) ===")
    print(
        classification_report(
            y_true_code[mask],
            y_pred_code[mask],
            labels=labels_wo_other,
            target_names=target_names_wo_other,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true_code,
        y_pred_code,
        labels=list(config.valid_labels),
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{config.class_names[c]}" for c in config.valid_labels],
        columns=[f"pred_{config.class_names[c]}" for c in config.valid_labels],
    )

    return cm_df

def plot_history(history_df):
    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss por epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # F1 macro con todas las clases
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_f1_macro_all"], marker="o", label="train_f1_macro_all")
    plt.plot(history_df["epoch"], history_df["val_f1_macro_all"], marker="o", label="val_f1_macro_all")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.title("F1 Macro (todas las clases)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # F1 macro excluyendo clase 8
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_f1_macro_excl_other"], marker="o", label="train_f1_macro_excl_other")
    plt.plot(history_df["epoch"], history_df["val_f1_macro_excl_other"], marker="o", label="val_f1_macro_excl_other")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.title("F1 Macro (excluyendo 'Otros')")
    plt.legend()
    plt.grid(True)
    plt.show()