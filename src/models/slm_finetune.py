import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Tuple
from typing import Optional
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.labels_climate import ID2LABEL, LABEL2ID

# Configuration Small Language Model - SLM  fine-tuning for climate fallacy multi-class task
@dataclass
class SLMConfigMC:
    model_name: str = "distilroberta-base"
    num_labels: int = len(ID2LABEL)
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    weight_decay: float = 0.01
    output_dir: str = "outputs/slm_climate_multiclass"

# Compute accuracy and macro/weighted F1 for evaluation
def compute_metrics_mc(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
#   Custom Trainer that applies class-weighted CrossEntropyLoss to handle label imbalance.
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # Don't pass labels twice to the model
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_slm_multiclass(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    cfg: SLMConfigMC,
) -> Trainer:
    # Attempts to drop rows with missing text/label and coerce to string
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

    #  Train/validation split (stratified)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[label_col],
        random_state=42,
    )

    # Keep only needed columns and drop index
    train_df = train_df[[text_col, label_col]].reset_index(drop=True)
    val_df = val_df[[text_col, label_col]].reset_index(drop=True)

    # Build HF Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Tokenize
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # Rename label column to "labels"
    train_ds = train_ds.rename_column(label_col, "labels")
    val_ds = val_ds.rename_column(label_col, "labels")

    # Remove original text column (no errors kwarg)
    cols_to_remove_train = [c for c in [text_col, "__index_level_0__"] if c in train_ds.column_names]
    cols_to_remove_val = [c for c in [text_col, "__index_level_0__"] if c in val_ds.column_names]

    train_ds = train_ds.remove_columns(cols_to_remove_train)
    val_ds = val_ds.remove_columns(cols_to_remove_val)

    # Set format for PyTorch
    train_ds.set_format("torch")
    val_ds.set_format("torch")

#  Computes class weights from the *training* splits
    labels_array = train_df[label_col].values
    classes = np.unique(labels_array)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_array,
    )

    class_weights = torch.tensor(class_weights_np, dtype=torch.float)
    print("Class weights (by label_id):", dict(zip(classes, class_weights_np)))

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.config.problem_type = "single_label_classification"

    # Training arguments
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
    )

    
    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mc,
        class_weights=class_weights,
    )

    trainer.train()
    return trainer
