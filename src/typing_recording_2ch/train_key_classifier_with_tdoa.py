# train_key_classifier_with_tdoa.py
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import (
    butter,
    sosfilt,
    sosfiltfilt,
    hilbert,
    find_peaks,
)

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
# 必要に応じて変更してください
# 例: KEY_LABELS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
KEY_LABELS = ["E", "H", "I", "L", "O", "P", "R", "SP", "T", "W", "Y"]

# 学習済みモデル保存先
MODEL_PATH = Path("typewriter_key_classifier.joblib")

# 音声読み込み時のサンプリング周波数
# 元データが96kHzなら 96000 のままでOK
TARGET_SR = 96000

# 既存の音響特徴量用の切り出し長
# CH1の音響解析にのみ使用する
CLIP_SEC = 0.075

# 音声波形を最大振幅位置で中央寄せする
# CH1の音響解析にのみ使用する
USE_PEAK_CENTERING = True

# テストデータ割合
TEST_SIZE = 0.2

# 乱数固定
RANDOM_STATE = 42


# ============================================================
# TDoA設定
# ============================================================

# TDoA特徴量を使うか
USE_TDOA_FEATURES = True

# マイク間距離[m]
MIC_DISTANCE_M = 0.21

# 音速[m/s]
SOUND_SPEED_M_S = 343.0

# TDoAの信頼度しきい値
# best_corrがこれ未満なら、tdoa_feature_msを0にする
TDOA_RELIABLE_CORR_THRESHOLD = 0.45

# 2つ目イベントを探し始める時刻[ms]
# 録音開始から120ms以降に2つ目の山が来る想定
TDOA_SECOND_SEARCH_START_MS = 120.0

# イベント検出用帯域
TDOA_EVENT_BAND_HZ = (300.0, 20000.0)

# 2つ目イベント後半の高域ピーク検出用帯域
TDOA_HIGH_BAND_HZ = (3000.0, 20000.0)

# 時間差計算用帯域
# 過去の解析では 500〜4000Hz の包絡線が比較的安定
TDOA_CALC_BAND_HZ = (500.0, 4000.0)

# 後半ピークの直前を使う
TDOA_PRE_MS = 3.0
TDOA_POST_MS = 0.2

# 包絡線の平滑化幅
TDOA_ENVELOPE_SMOOTH_MS = 0.20

# 2つ目イベント内で後半ピークを選ぶときの比率
TDOA_TAIL_PEAK_RATIO = 0.45

# イベントピーク検出の最小間隔
TDOA_MIN_EVENT_DISTANCE_MS = 50.0


TDOA_FEATURE_NAMES = [
    "tdoa_feature_ms",
    "tdoa_reliable",
    "best_corr",
    "energy_CH1_over_CH2_dB",
    "abs_tdoa_feature_ms",
]


# ============================================================
# 音声前処理
# ============================================================

def load_audio_stereo(path: Path, target_sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    2ch WAVを読み込む。

    戻り値:
        ch1: 左マイク
        ch2: 右マイク

    注意:
        既存の音響特徴量にはch1のみを使う。
        TDoAにはch1/ch2の両方を使う。
    """
    y, sr = librosa.load(path, sr=target_sr, mono=False)

    y = np.nan_to_num(y)

    if y.ndim == 1:
        # モノラルの場合の保険
        ch1 = y.astype(np.float32)
        ch2 = np.zeros_like(ch1)
        return ch1, ch2

    # librosaのmulti-channelは shape = (channels, samples)
    ch1 = y[0].astype(np.float32)

    if y.shape[0] >= 2:
        ch2 = y[1].astype(np.float32)
    else:
        ch2 = np.zeros_like(ch1)

    return ch1, ch2


def peak_center_audio(y: np.ndarray, sr: int, clip_sec: float) -> np.ndarray:
    """
    最大振幅位置を中心に、固定長で切り出す。
    足りない部分はゼロ埋めする。

    注意:
        これはCH1の音響特徴量用。
        TDoA計算には元の2ch波形を使う。
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
# 既存の音響特徴量抽出
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


def extract_audio_features_from_ch1(y: np.ndarray, sr: int) -> np.ndarray:
    """
    CH1の音声波形から分類用特徴量を抽出する。
    ここは既存処理と同等。
    """
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


# ============================================================
# TDoA特徴量抽出
# ============================================================

def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """
    バンドパスフィルタ。
    短すぎる波形ではfiltfiltが失敗することがあるため、
    その場合は片方向フィルタにフォールバックする。
    """
    if len(x) == 0:
        return x

    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)

    if low >= high:
        return np.zeros_like(x)

    sos = butter(order, [low, high], btype="bandpass", output="sos")

    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        return sosfilt(sos, x)


def smooth_moving_average(x: np.ndarray, sr: int, smooth_ms: float) -> np.ndarray:
    n = max(1, int(round(sr * smooth_ms / 1000.0)))

    if n <= 1:
        return x

    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def envelope_band(
    x: np.ndarray,
    sr: int,
    band_hz: tuple[float, float],
    smooth_ms: float,
) -> np.ndarray:
    """
    指定帯域のヒルベルト包絡線を作る。
    """
    if len(x) == 0:
        return x

    y = bandpass(x, sr, band_hz[0], band_hz[1])
    env = np.abs(hilbert(y))
    env = smooth_moving_average(env, sr, smooth_ms)

    return env


def normalize_for_corr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    std = np.std(x)
    if std < 1e-12:
        return x * 0.0

    return x / std


def corr_at_lag(ch1_feature: np.ndarray, ch2_feature: np.ndarray, lag: int) -> float:
    """
    lagの定義:
        lag = t_CH2 - t_CH1 のサンプル数

    lag > 0:
        CH2がCH1より遅い
        つまりCH1、左マイクが先に鳴った

    lag < 0:
        CH2がCH1より早い
        つまりCH2、右マイクが先に鳴った
    """
    if lag > 0:
        a = ch1_feature[:-lag]
        b = ch2_feature[lag:]
    elif lag < 0:
        a = ch1_feature[-lag:]
        b = ch2_feature[:lag]
    else:
        a = ch1_feature
        b = ch2_feature

    if len(a) < 8:
        return np.nan

    a = normalize_for_corr(a)
    b = normalize_for_corr(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return np.nan

    return float(np.dot(a, b) / denom)


def estimate_lag_by_xcorr(
    ch1_feature: np.ndarray,
    ch2_feature: np.ndarray,
    sr: int,
    max_lag_s: float,
) -> tuple[float, float]:
    """
    相互相関でTDoAを推定する。

    戻り値:
        lag_samples:
            t_CH2 - t_CH1 のサンプル数
        best_corr:
            最大相関値
    """
    if len(ch1_feature) < 16 or len(ch2_feature) < 16:
        return np.nan, np.nan

    max_lag_samples = int(round(max_lag_s * sr))
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)

    corrs = np.array([
        corr_at_lag(ch1_feature, ch2_feature, int(lag))
        for lag in lags
    ])

    if np.all(np.isnan(corrs)):
        return np.nan, np.nan

    best_i = int(np.nanargmax(corrs))
    best_lag = float(lags[best_i])
    best_corr = float(corrs[best_i])

    # サブサンプル推定
    # 相関ピークの前後3点を放物線近似して、整数サンプル未満のズレを推定する
    if 0 < best_i < len(corrs) - 1:
        y0 = corrs[best_i - 1]
        y1 = corrs[best_i]
        y2 = corrs[best_i + 1]

        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12 and not np.isnan(denom):
            offset = 0.5 * (y0 - y2) / denom
            offset = float(np.clip(offset, -1.0, 1.0))
            best_lag += offset

    return best_lag, best_corr


def find_second_event_peak(event_env: np.ndarray, sr: int) -> int:
    """
    大まかな2つ目イベント位置を探す。
    """
    if len(event_env) == 0:
        return 0

    max_val = float(np.max(event_env))
    if max_val <= 0:
        return int(len(event_env) // 2)

    distance = int(round(sr * TDOA_MIN_EVENT_DISTANCE_MS / 1000.0))
    height = max_val * 0.12
    prominence = max_val * 0.06

    peaks, _ = find_peaks(
        event_env,
        distance=distance,
        height=height,
        prominence=prominence,
    )

    second_start = int(round(sr * TDOA_SECOND_SEARCH_START_MS / 1000.0))

    # second_search_start以降のピークを優先
    peaks_after = peaks[peaks >= second_start]
    if len(peaks_after) > 0:
        return int(peaks_after[0])

    # 見つからなければ、second_start以降の最大値
    if second_start < len(event_env):
        return int(second_start + np.argmax(event_env[second_start:]))

    # WAVが短い場合は全体から最大値を使う
    return int(np.argmax(event_env))


def find_event_region(
    env: np.ndarray,
    peak_idx: int,
    sr: int,
    threshold_ratio: float = 0.12,
) -> tuple[int, int]:
    """
    指定ピークの周辺で、包絡線がしきい値以上の領域を取得する。
    """
    if len(env) == 0:
        return 0, 0

    peak_idx = int(np.clip(peak_idx, 0, len(env) - 1))
    peak_val = float(env[peak_idx])

    if peak_val <= 0:
        return 0, len(env)

    th = peak_val * threshold_ratio

    left = peak_idx
    while left > 1 and env[left] > th:
        left -= 1

    right = peak_idx
    while right < len(env) - 2 and env[right] > th:
        right += 1

    # 領域が狭すぎる場合の保険
    min_half = int(round(sr * 8.0 / 1000.0))
    if right - left < min_half:
        left = max(0, peak_idx - min_half)
        right = min(len(env), peak_idx + min_half)

    return left, right


def find_tail_high_peak(
    hf_env_mix: np.ndarray,
    region_start: int,
    region_end: int,
    sr: int,
) -> int:
    """
    2つ目イベント領域の中から、後半側の高域ピークを選ぶ。

    狙い:
        立ち上がり側のキャリッジ音ではなく、
        終わり側のアーム戻り音を拾う。
    """
    if len(hf_env_mix) == 0:
        return 0

    region_start = int(np.clip(region_start, 0, len(hf_env_mix) - 1))
    region_end = int(np.clip(region_end, region_start + 1, len(hf_env_mix)))

    seg = hf_env_mix[region_start:region_end]

    if len(seg) < 8:
        return int(region_start + np.argmax(seg))

    max_val = float(np.max(seg))
    if max_val <= 0:
        return int(region_start + np.argmax(seg))

    # 細かい振動を拾いすぎないように0.4ms程度は離す
    distance = max(1, int(round(sr * 0.4 / 1000.0)))

    peaks, props = find_peaks(
        seg,
        height=max_val * TDOA_TAIL_PEAK_RATIO,
        distance=distance,
    )

    if len(peaks) > 0:
        # 条件を満たすピークのうち最後を使う
        return int(region_start + peaks[-1])

    # 見つからなければ、領域内の後半半分の最大値
    mid = len(seg) // 2
    return int(region_start + mid + np.argmax(seg[mid:]))


def calc_tdoa_features(ch1: np.ndarray, ch2: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """
    CH1/CH2からTDoA特徴量を計算する。

    学習に入れる特徴量:
        tdoa_feature_ms:
            信頼できる場合のみ生TDoAを入れる。
            怪しい場合は0にする。

        tdoa_reliable:
            TDoAを信用するかどうかの0/1フラグ。

        best_corr:
            相互相関の最大値。
            TDoA計算の信頼度目安。

        energy_CH1_over_CH2_dB:
            TDoA計算窓内の左右エネルギー比。

        abs_tdoa_feature_ms:
            中央からどの程度離れていそうかの補助特徴量。
    """
    max_lag_s = MIC_DISTANCE_M / SOUND_SPEED_M_S

    # 長さを合わせる
    n = min(len(ch1), len(ch2))
    ch1 = ch1[:n]
    ch2 = ch2[:n]

    if n < 32:
        info = {
            "tdoa_raw_ms_CH2_minus_CH1": np.nan,
            "tdoa_feature_ms": 0.0,
            "tdoa_reliable": 0,
            "best_corr": 0.0,
            "energy_CH1_over_CH2_dB": 0.0,
            "abs_tdoa_feature_ms": 0.0,
            "second_event_peak_ms": np.nan,
            "tail_peak_ms": np.nan,
            "tdoa_window_start_ms": np.nan,
            "tdoa_window_end_ms": np.nan,
        }
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0]), info

    # イベント検出用包絡線
    env1_event = envelope_band(
        ch1,
        sr,
        TDOA_EVENT_BAND_HZ,
        TDOA_ENVELOPE_SMOOTH_MS,
    )
    env2_event = envelope_band(
        ch2,
        sr,
        TDOA_EVENT_BAND_HZ,
        TDOA_ENVELOPE_SMOOTH_MS,
    )
    event_env = 0.5 * (env1_event + env2_event)

    # 2つ目イベントを探す
    second_peak = find_second_event_peak(event_env, sr)
    region_start, region_end = find_event_region(event_env, second_peak, sr)

    # 高域包絡線で2つ目イベント後半ピークを探す
    env1_hf = envelope_band(
        ch1,
        sr,
        TDOA_HIGH_BAND_HZ,
        TDOA_ENVELOPE_SMOOTH_MS,
    )
    env2_hf = envelope_band(
        ch2,
        sr,
        TDOA_HIGH_BAND_HZ,
        TDOA_ENVELOPE_SMOOTH_MS,
    )
    hf_env_mix = 0.5 * (env1_hf + env2_hf)

    tail_peak = find_tail_high_peak(
        hf_env_mix,
        region_start,
        region_end,
        sr,
    )

    # TDoA計算用の低〜中域包絡線
    env1_tdoa = envelope_band(
        ch1,
        sr,
        TDOA_CALC_BAND_HZ,
        TDOA_ENVELOPE_SMOOTH_MS,
    )
    env2_tdoa = envelope_band(
        ch2,
        sr,
        TDOA_CALC_BAND_HZ,
        TDOA_ENVELOPE_SMOOTH_MS,
    )

    pre = int(round(sr * TDOA_PRE_MS / 1000.0))
    post = int(round(sr * TDOA_POST_MS / 1000.0))

    win_start = max(0, tail_peak - pre)
    win_end = min(n, tail_peak + post)

    f1 = env1_tdoa[win_start:win_end]
    f2 = env2_tdoa[win_start:win_end]

    lag_samples, best_corr = estimate_lag_by_xcorr(
        f1,
        f2,
        sr=sr,
        max_lag_s=max_lag_s,
    )

    if np.isnan(lag_samples):
        tdoa_raw_ms = np.nan
    else:
        tdoa_raw_ms = lag_samples / sr * 1000.0

    if np.isnan(best_corr):
        best_corr_clean = 0.0
    else:
        best_corr_clean = float(best_corr)

    # 左右エネルギー比
    eps = 1e-12
    if len(f1) > 0 and len(f2) > 0:
        rms1 = float(np.sqrt(np.mean(f1 ** 2) + eps))
        rms2 = float(np.sqrt(np.mean(f2 ** 2) + eps))
        energy_lr_db = float(20.0 * np.log10((rms1 + eps) / (rms2 + eps)))
    else:
        energy_lr_db = 0.0

    # 信頼度判定
    if np.isnan(tdoa_raw_ms):
        tdoa_reliable = 0
        tdoa_feature_ms = 0.0
    elif best_corr_clean >= TDOA_RELIABLE_CORR_THRESHOLD:
        tdoa_reliable = 1
        tdoa_feature_ms = float(tdoa_raw_ms)
    else:
        tdoa_reliable = 0
        tdoa_feature_ms = 0.0

    abs_tdoa_feature_ms = abs(tdoa_feature_ms)

    tdoa_features = np.array([
        tdoa_feature_ms,
        float(tdoa_reliable),
        best_corr_clean,
        energy_lr_db,
        abs_tdoa_feature_ms,
    ], dtype=np.float64)

    info = {
        "tdoa_raw_ms_CH2_minus_CH1": tdoa_raw_ms,
        "tdoa_feature_ms": tdoa_feature_ms,
        "tdoa_reliable": tdoa_reliable,
        "best_corr": best_corr_clean,
        "energy_CH1_over_CH2_dB": energy_lr_db,
        "abs_tdoa_feature_ms": abs_tdoa_feature_ms,
        "second_event_peak_ms": second_peak / sr * 1000.0,
        "tail_peak_ms": tail_peak / sr * 1000.0,
        "tdoa_window_start_ms": win_start / sr * 1000.0,
        "tdoa_window_end_ms": win_end / sr * 1000.0,
    }

    return tdoa_features, info


# ============================================================
# ファイル単位の特徴量抽出
# ============================================================

def extract_features_from_file(
    path: Path,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    1つのWAVファイルから特徴量を抽出する。

    音響特徴量:
        CH1のみ使用

    TDoA特徴量:
        CH1/CH2を使用
    """
    ch1, ch2 = load_audio_stereo(path, TARGET_SR)

    # 既存の音響解析にはCH1のみを使う
    y_for_audio_features = ch1.copy()

    if USE_PEAK_CENTERING:
        y_for_audio_features = peak_center_audio(
            y_for_audio_features,
            TARGET_SR,
            CLIP_SEC,
        )

    audio_features = extract_audio_features_from_ch1(
        y_for_audio_features,
        TARGET_SR,
    )

    if USE_TDOA_FEATURES:
        tdoa_features, tdoa_info = calc_tdoa_features(ch1, ch2, TARGET_SR)
        features = np.concatenate([audio_features, tdoa_features])
    else:
        tdoa_info = {}
        features = audio_features

    if return_info:
        return features, tdoa_info

    return features


# ============================================================
# データセット読み込み
# ============================================================

def load_dataset():
    X = []
    y = []
    paths = []
    tdoa_rows = []

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
                feat, tdoa_info = extract_features_from_file(
                    wav_path,
                    return_info=True,
                )
            except Exception as e:
                print(f"[ERROR] 特徴量抽出失敗: {wav_path}")
                print(e)
                continue

            X.append(feat)
            y.append(label)
            paths.append(wav_path)

            row = {
                "label": label,
                "file": str(wav_path),
            }
            row.update(tdoa_info)
            tdoa_rows.append(row)

    X = np.array(X)
    y = np.array(y)

    tdoa_df = pd.DataFrame(tdoa_rows)
    tdoa_df.to_csv("tdoa_features.csv", index=False, encoding="utf-8-sig")
    print("Saved TDoA feature log: tdoa_features.csv")

    return X, y, paths, tdoa_df


# ============================================================
# 学習・評価
# ============================================================

def train_and_evaluate():
    print("=== Load Dataset ===")
    X, y, paths, tdoa_df = load_dataset()

    print()
    print("=== Dataset Info ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print(pd.DataFrame({
        "label": unique,
        "count": counts,
    }))

    if USE_TDOA_FEATURES:
        print()
        print("=== TDoA Feature Summary ===")
        cols = [
            "label",
            "tdoa_raw_ms_CH2_minus_CH1",
            "tdoa_feature_ms",
            "tdoa_reliable",
            "best_corr",
            "energy_CH1_over_CH2_dB",
        ]
        available_cols = [c for c in cols if c in tdoa_df.columns]
        print(tdoa_df[available_cols].groupby("label").agg(["mean", "std", "min", "max"]))

        reliable_rate = tdoa_df.groupby("label")["tdoa_reliable"].mean()
        print()
        print("TDoA reliable rate:")
        print(reliable_rate)

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
        "use_tdoa_features": USE_TDOA_FEATURES,
        "tdoa_feature_names": TDOA_FEATURE_NAMES,
        "mic_distance_m": MIC_DISTANCE_M,
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "tdoa_reliable_corr_threshold": TDOA_RELIABLE_CORR_THRESHOLD,
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

    feat, tdoa_info = extract_features_from_file(
        wav_path,
        return_info=True,
    )
    X_one = feat.reshape(1, -1)

    pred = model.predict(X_one)[0]

    print()
    print("=== TDoA info ===")
    for k, v in tdoa_info.items():
        print(f"{k}: {v}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_one)[0]
        classes = model.classes_

        ranking = sorted(
            zip(classes, proba),
            key=lambda x: x[1],
            reverse=True,
        )

        print()
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
    # predict_wav("recorded_triggers/Q/sample001.wav")