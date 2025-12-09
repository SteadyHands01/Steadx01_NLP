from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


@dataclass
class BaselineConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 20000
    ngram_range: tuple = (1, 2)


def run_baseline_multiclass(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    cfg: BaselineConfig,
) -> Tuple[LinearSVC, TfidfVectorizer]:
    # 🔐 Safety: drop rows with missing text or label
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].fillna("").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[label_col],
    )

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("=== Baseline (TF-IDF + Linear SVM) ===")
    print(classification_report(y_test, y_pred))

    return model, vectorizer