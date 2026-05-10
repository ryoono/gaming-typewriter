import warnings
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 設定
# ============================================================

# データセットのルートフォルダ
DATASET_DIR = Path("recorded_triggers")

# 分類対象キー
KEY_LABELS = ["D", "E", "H", "L", "O", "R", "W"]

# 学習済みモデル保存先
MODEL_PATH = Path("typewriter_key_classifier.joblib")

# 音声読み込み時のサンプリング周波数
# 元データが96kHzなら 96000 のままでOK
TARGET_SR = 96000

# トリガ前後100msで保存している前提なら0.2秒
CLIP_SEC = 0.075

# 音声波形を最大振幅位置で中央寄せする
# トリガ位置に多少ズレがある場合に有効
USE_PEAK_CENTERING = True

# テストデータ割合
TEST_SIZE = 0.2

# 乱数固定
RANDOM_STATE = 42


# ============================================================
# 音声前処理
# ============================================================

def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """
    wavをモノラルfloat配列として読み込む。
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)

    # NaN対策
    y = np.nan_to_num(y)

    return y


def peak_center_audio(y: np.ndarray, sr: int, clip_sec: float) -> np.ndarray:
    """
    最大振幅位置を中心に、固定長で切り出す。
    足りない部分はゼロ埋めする。
    """
    target_len = int(sr * clip_sec)

    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)

    peak_index = int(np.argmax(np.abs(y)))
    half_len = target_len // 2

    start = peak_index - half_len
    end = start + target_len

    output = np.zeros(target_len, dtype=np.float32)

    src_start = max(0, start)
    src_end = min(len(y), end)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        output[dst_start:dst_end] = y[src_start:src_end]

    return output


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    音量差が大きすぎる場合に備えて正規化する。
    ただし、完全に無音ならそのまま返す。
    """
    max_abs = np.max(np.abs(y))

    if max_abs < 1e-9:
        return y

    return y / max_abs


# ============================================================
# 特徴量抽出
# ============================================================

def calc_stats(feature_matrix: np.ndarray) -> np.ndarray:
    """
    時間方向を持つ特徴量から、平均・標準偏差・最大・最小を作る。
    shape: (features, frames)
    """
    return np.concatenate([
        np.mean(feature_matrix, axis=1),
        np.std(feature_matrix, axis=1),
        np.max(feature_matrix, axis=1),
        np.min(feature_matrix, axis=1),
    ])


def band_energy_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    FFTから周波数帯域ごとのエネルギー比を抽出する。
    タイプライターの打鍵音は高周波成分にも差が出る可能性があるため、
    MFCCとは別に帯域エネルギーも特徴量に入れる。
    """
    if len(y) == 0:
        return np.zeros(10)

    window = np.hanning(len(y))
    spectrum = np.abs(np.fft.rfft(y * window)) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    total_energy = np.sum(spectrum) + 1e-12

    bands = [
        (0, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 12000),
        (12000, 16000),
        (16000, 24000),
        (24000, 32000),
        (32000, sr // 2),
    ]

    features = []

    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energy = np.sum(spectrum[mask]) / total_energy
        features.append(np.log10(energy + 1e-12))

    return np.array(features)


def extract_features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    1つの音声波形から分類用特徴量を抽出する。
    """
    # 解析用に正規化
    y = normalize_audio(y)

    n_fft = 2048
    hop_length = 512

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(mfcc))

    # MFCCの時間変化
    mfcc_delta = librosa.feature.delta(mfcc)
    features.append(calc_stats(mfcc_delta))

    # RMS
    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(zcr))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(bandwidth))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(rolloff))

    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    features.append(calc_stats(flatness))

    # FFT帯域エネルギー
    features.append(band_energy_features(y, sr))

    return np.concatenate(features)


def extract_features_from_file(path: Path) -> np.ndarray:
    y = load_audio_mono(path, TARGET_SR)

    if USE_PEAK_CENTERING:
        y = peak_center_audio(y, TARGET_SR, CLIP_SEC)

    return extract_features_from_audio(y, TARGET_SR)


# ============================================================
# データセット読み込み
# ============================================================

def load_dataset():
    X = []
    y = []
    paths = []

    for label in KEY_LABELS:
        label_dir = DATASET_DIR / label

        if not label_dir.exists():
            raise FileNotFoundError(f"フォルダが見つかりません: {label_dir}")

        wav_files = sorted(label_dir.glob("*.wav"))

        if len(wav_files) == 0:
            raise FileNotFoundError(f"wavファイルがありません: {label_dir}")

        print(f"{label}: {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                feat = extract_features_from_file(wav_path)
            except Exception as e:
                print(f"[ERROR] 特徴量抽出失敗: {wav_path}")
                print(e)
                continue

            X.append(feat)
            y.append(label)
            paths.append(wav_path)

    X = np.array(X)
    y = np.array(y)

    return X, y, paths


# ============================================================
# 学習・評価
# ============================================================

def train_and_evaluate():
    print("=== Load Dataset ===")
    X, y, paths = load_dataset()

    print()
    print("=== Dataset Info ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print(pd.DataFrame({
        "label": unique,
        "count": counts,
    }))

    print()
    print("=== Train/Test Split ===")

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"train: {len(y_train)}")
    print(f"test : {len(y_test)}")

    # 標準化 + SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

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
    print("=== Training ===")
    model.fit(X_train, y_train)

    print()
    print("=== Test Evaluation ===")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=KEY_LABELS))

    print()
    print("Misclassified files:")
    miss_count = 0
    for true_label, pred_label, wav_path in zip(y_test, y_pred, paths_test):
        if true_label != pred_label:
            miss_count += 1
            print(f"  true={true_label}, pred={pred_label}, file={wav_path}")

    if miss_count == 0:
        print("  None")

    # 混同行列
    cm = confusion_matrix(y_test, y_pred, labels=KEY_LABELS)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=KEY_LABELS,
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix / Accuracy={acc:.3f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.show()

    # 全データで再学習して保存
    print()
    print("=== Train Final Model with All Data ===")
    model.fit(X, y)

    joblib.dump({
        "model": model,
        "labels": KEY_LABELS,
        "target_sr": TARGET_SR,
        "clip_sec": CLIP_SEC,
        "use_peak_centering": USE_PEAK_CENTERING,
    }, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")

    return model


# ============================================================
# 推論
# ============================================================

def predict_wav(wav_path: str | Path, model_path: str | Path = MODEL_PATH):
    wav_path = Path(wav_path)

    saved = joblib.load(model_path)
    model = saved["model"]

    feat = extract_features_from_file(wav_path)
    X_one = feat.reshape(1, -1)

    pred = model.predict(X_one)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_one)[0]
        classes = model.classes_

        ranking = sorted(
            zip(classes, proba),
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"file: {wav_path}")
        print(f"predict: {pred}")
        print("probability:")
        for label, p in ranking:
            print(f"  {label}: {p:.3f}")
    else:
        print(f"file: {wav_path}")
        print(f"predict: {pred}")

    return pred


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    train_and_evaluate()

    # 推論例
    # predict_wav("dataset/D/sample001.wav")