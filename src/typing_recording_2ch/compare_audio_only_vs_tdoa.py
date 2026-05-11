# compare_audio_only_vs_tdoa.py
# -*- coding: utf-8 -*-

"""
TDoAあり / TDoAなし の精度比較用スクリプト

前提:
- 同じフォルダに train_key_classifier_with_tdoa.py がある
- recorded_triggers/ 配下に各キーのWAVフォルダがある
- train_key_classifier_with_tdoa.py 側の KEY_LABELS は設定済み

実行:
    python compare_audio_only_vs_tdoa.py
"""

import warnings
from pathlib import Path

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 既存の学習スクリプトを読み込む
base = importlib.import_module("train_key_classifier_with_tdoa")


OUTPUT_DIR = Path("compare_tdoa_result")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = base.RANDOM_STATE
TEST_SIZE = base.TEST_SIZE
KEY_LABELS = base.KEY_LABELS


def make_model():
    """
    既存コードと同じSVMモデルを作る。
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])


def load_features(use_tdoa: bool):
    """
    既存コードの USE_TDOA_FEATURES を切り替えて特徴量を読み込む。
    """
    print()
    print("=" * 70)
    print(f"Load dataset: USE_TDOA_FEATURES = {use_tdoa}")
    print("=" * 70)

    base.USE_TDOA_FEATURES = use_tdoa

    X, y, paths, tdoa_df = base.load_dataset()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, paths, tdoa_df


def evaluate_model(name: str, X, y, paths, split_indices):
    """
    同じ分割で学習・評価する。
    """
    print()
    print("=" * 70)
    print(f"Evaluate: {name}")
    print("=" * 70)

    train_idx, test_idx = split_indices

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    paths_test = [paths[i] for i in test_idx]

    model = make_model()

    print()
    print("=== Cross Validation ===")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

    print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("CV scores  :", np.round(cv_scores, 3))

    print()
    print("=== Train/Test Evaluation ===")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}")
    print()
    print("Classification report:")
    report_text = classification_report(
        y_test,
        y_pred,
        labels=KEY_LABELS,
    )
    print(report_text)

    print()
    print("Misclassified files:")
    miss_rows = []

    for true_label, pred_label, wav_path in zip(y_test, y_pred, paths_test):
        if true_label != pred_label:
            row = {
                "mode": name,
                "true": true_label,
                "pred": pred_label,
                "file": str(wav_path),
            }
            miss_rows.append(row)
            print(f"  true={true_label}, pred={pred_label}, file={wav_path}")

    if len(miss_rows) == 0:
        print("  None")

    # 混同行列
    cm = confusion_matrix(y_test, y_pred, labels=KEY_LABELS)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=KEY_LABELS,
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} / Accuracy={acc:.3f}")
    plt.tight_layout()

    png_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved: {png_path}")

    result = {
        "mode": name,
        "n_features": X.shape[1],
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float(acc),
        "miss_count": int(len(miss_rows)),
    }

    return result, miss_rows


def main():
    # 1. TDoAなし特徴量
    X_audio, y_audio, paths_audio, _ = load_features(use_tdoa=False)

    # 2. TDoAあり特徴量
    X_tdoa, y_tdoa, paths_tdoa, tdoa_df = load_features(use_tdoa=True)

    # 念のため、ラベル順とファイル順が一致しているか確認
    if not np.array_equal(y_audio, y_tdoa):
        raise RuntimeError("TDoAあり/なしでラベル順が一致していません。")

    if [str(p) for p in paths_audio] != [str(p) for p in paths_tdoa]:
        raise RuntimeError("TDoAあり/なしでファイル順が一致していません。")

    y = y_audio
    paths = paths_audio

    print()
    print("=" * 70)
    print("Dataset summary")
    print("=" * 70)

    unique, counts = np.unique(y, return_counts=True)
    print(pd.DataFrame({
        "label": unique,
        "count": counts,
    }))

    # 同じ train/test split を使うため、indexを分割する
    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    split_indices = (train_idx, test_idx)

    results = []
    all_miss_rows = []

    # TDoAなし
    result_audio, miss_audio = evaluate_model(
        name="audio_only",
        X=X_audio,
        y=y_audio,
        paths=paths_audio,
        split_indices=split_indices,
    )
    results.append(result_audio)
    all_miss_rows.extend(miss_audio)

    # TDoAあり
    result_tdoa, miss_tdoa = evaluate_model(
        name="with_tdoa",
        X=X_tdoa,
        y=y_tdoa,
        paths=paths_tdoa,
        split_indices=split_indices,
    )
    results.append(result_tdoa)
    all_miss_rows.extend(miss_tdoa)

    # 結果保存
    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    miss_df = pd.DataFrame(all_miss_rows)
    miss_path = OUTPUT_DIR / "misclassified_files.csv"
    miss_df.to_csv(miss_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print("Comparison summary")
    print("=" * 70)
    print(summary_df)

    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {miss_path}")

    # TDoAログも保存しておく
    if tdoa_df is not None and len(tdoa_df) > 0:
        tdoa_log_path = OUTPUT_DIR / "tdoa_features.csv"
        tdoa_df.to_csv(tdoa_log_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {tdoa_log_path}")


if __name__ == "__main__":
    main()