# realtime_typewriter_predict.py
# -*- coding: utf-8 -*-

"""
タイプライター打鍵音 リアルタイム推論プログラム

仕様:
- Arduinoからシリアルで "1" を受信したらトリガ
- トリガ前後の音声をリングバッファから取得
- CH1のみを使って音響特徴量を抽出
- 学習済みSVMモデルを読み込んで推論
- 標準出力には推論した文字のみを出力する

例:
    タイプライターで hello typewriter と打つ
    コマンドラインにも hello typewriter と表示される

注意:
- TDoAは使わない
- 2ch/4ch入力デバイスでも、推論にはCH1のみ使用する
- ログを出すと文字列出力を邪魔するため、デフォルトでは標準出力にログを出さない
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import joblib
import librosa
import numpy as np
import serial
import sounddevice as sd

from sklearn.pipeline import Pipeline


# ============================================================
# 設定
# ============================================================

# 学習済みモデル
# TDoAなし、音響特徴量のみで学習したモデルを指定してください
MODEL_PATH = Path("typewriter_key_classifier_audio_only.joblib")

# ArduinoのCOMポート
SERIAL_PORT = "COM7"

# Arduino側の Serial.begin() と合わせる
SERIAL_BAUDRATE = 1000000

# マイク設定
DEVICE_ID = 12
SAMPLERATE = 96000

# ZOOM AMS-44 ASIO Driverの都合で4ch入力する場合は4
# 本当に2ch入力なら2、1ch入力なら1でもOK
INPUT_CHANNELS = 4

# 推論に使うチャンネル
# 0: CH1, 1: CH2, ...
PREDICT_CHANNEL_INDEX = 0

# sounddeviceの録音データ型
DTYPE = "int32"

# トリガ前後の取得時間
# 学習時の録音条件に合わせる
PRE_TRIGGER_SEC = 0.015
POST_TRIGGER_SEC = 0.400

# リングバッファ長
RING_BUFFER_SEC = 5.0

# トリガ後音声待ちタイムアウト
AUDIO_WAIT_TIMEOUT_SEC = 1.0

# 学習時の特徴量抽出設定
# 学習コードと合わせる
CLIP_SEC = 0.075
USE_PEAK_CENTERING = True

# 表示設定
# True: E -> e, H -> h のように小文字で出力
# False: 学習ラベルそのまま出力
OUTPUT_LOWERCASE = True

# スペースキーのラベル
SPACE_LABELS = {"SP", "SPACE", "Space", "space", " "}

# デバッグログ
# 標準出力には推論文字だけを出したいので、通常はFalse
# Trueにするとstderrにログを出す
VERBOSE = False


# ============================================================
# ログ
# ============================================================

def log(msg: str) -> None:
    if VERBOSE:
        print(msg, file=sys.stderr, flush=True)


# ============================================================
# 音声リングバッファ
# ============================================================

class AudioRingBuffer:
    def __init__(self, samplerate: int, channels: int, buffer_sec: float):
        self.samplerate = samplerate
        self.channels = channels
        self.buffer_samples = int(samplerate * buffer_sec)

        self.buffer = np.zeros(
            (self.buffer_samples, channels),
            dtype=np.int32,
        )

        self.write_pos = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        frames = len(data)

        if frames <= 0:
            return

        with self.lock:
            if frames >= self.buffer_samples:
                data = data[-self.buffer_samples:]
                frames = len(data)

            end_pos = self.write_pos + frames

            if end_pos <= self.buffer_samples:
                self.buffer[self.write_pos:end_pos] = data
            else:
                first_len = self.buffer_samples - self.write_pos
                second_len = frames - first_len

                self.buffer[self.write_pos:] = data[:first_len]
                self.buffer[:second_len] = data[first_len:]

            self.write_pos = (self.write_pos + frames) % self.buffer_samples
            self.total_written += frames

    def get_total_written(self) -> int:
        with self.lock:
            return self.total_written

    def read_range(self, start_abs: int, end_abs: int) -> np.ndarray | None:
        with self.lock:
            current_total = self.total_written
            oldest_available = max(0, current_total - self.buffer_samples)

            if start_abs < oldest_available:
                return None

            if end_abs > current_total:
                return None

            length = end_abs - start_abs
            if length <= 0:
                return None

            start_pos = start_abs % self.buffer_samples
            end_pos = start_pos + length

            if end_pos <= self.buffer_samples:
                audio = self.buffer[start_pos:end_pos].copy()
            else:
                first_len = self.buffer_samples - start_pos
                second_len = length - first_len

                audio = np.vstack([
                    self.buffer[start_pos:].copy(),
                    self.buffer[:second_len].copy(),
                ])

            return audio


audio_buffer = AudioRingBuffer(
    samplerate=SAMPLERATE,
    channels=INPUT_CHANNELS,
    buffer_sec=RING_BUFFER_SEC,
)


# ============================================================
# 音声入力コールバック
# ============================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        log(f"[Audio status] {status}")

    audio_buffer.write(indata.copy())


# ============================================================
# 音声前処理
# ============================================================

def int_audio_to_float(y: np.ndarray) -> np.ndarray:
    """
    sounddeviceのint32音声をfloatへ変換する。
    その後normalize_audio()も行うため、厳密なスケールよりも形が重要。
    """
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        scale = max(abs(info.min), abs(info.max))
        return y.astype(np.float32) / scale

    return y.astype(np.float32)


def peak_center_audio(y: np.ndarray, sr: int, clip_sec: float) -> np.ndarray:
    """
    最大振幅位置を中心に固定長で切り出す。
    学習時と同じ処理。
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
    学習時と同じ処理。
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
    CH1音声から分類用特徴量を抽出する。
    TDoAは使わない。
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

    # MFCC delta
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


def extract_features_from_audio_array(ch1_audio: np.ndarray) -> np.ndarray:
    """
    録音済みCH1配列から学習時と同じ特徴量を抽出する。
    """
    y = int_audio_to_float(ch1_audio)

    if USE_PEAK_CENTERING:
        y = peak_center_audio(y, SAMPLERATE, CLIP_SEC)

    feat = extract_audio_features_from_ch1(y, SAMPLERATE)

    return feat


# ============================================================
# モデル読み込み
# ============================================================

def load_model(model_path: Path):
    saved = joblib.load(model_path)

    if isinstance(saved, dict) and "model" in saved:
        model = saved["model"]

        saved_target_sr = saved.get("target_sr")
        saved_clip_sec = saved.get("clip_sec")
        saved_use_peak_centering = saved.get("use_peak_centering")

        if saved_target_sr is not None and saved_target_sr != SAMPLERATE:
            log(f"[WARN] model target_sr={saved_target_sr}, runtime SAMPLERATE={SAMPLERATE}")

        if saved_clip_sec is not None and abs(float(saved_clip_sec) - CLIP_SEC) > 1e-9:
            log(f"[WARN] model clip_sec={saved_clip_sec}, runtime CLIP_SEC={CLIP_SEC}")

        if saved_use_peak_centering is not None and bool(saved_use_peak_centering) != USE_PEAK_CENTERING:
            log(
                f"[WARN] model use_peak_centering={saved_use_peak_centering}, "
                f"runtime USE_PEAK_CENTERING={USE_PEAK_CENTERING}"
            )

        use_tdoa = saved.get("use_tdoa_features")
        if use_tdoa is True:
            log("[WARN] このモデルはTDoAありで学習されている可能性があります。")
            log("       本プログラムはTDoAなし194特徴量を入力します。")
    else:
        model = saved

    return model


def expected_feature_count(model) -> int | None:
    """
    学習済みモデルが期待する特徴量数を取得する。
    Pipelineの場合でも取得を試みる。
    """
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)

    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)

    return None


# ============================================================
# 推論
# ============================================================

def label_to_char(label) -> str:
    """
    学習ラベルを出力文字へ変換する。
    """
    s = str(label)

    if s in SPACE_LABELS:
        return " "

    if OUTPUT_LOWERCASE:
        return s.lower()

    return s


def predict_char(model, ch1_audio: np.ndarray) -> str:
    feat = extract_features_from_audio_array(ch1_audio)

    expected = expected_feature_count(model)
    if expected is not None and feat.shape[0] != expected:
        raise RuntimeError(
            f"特徴量数がモデルと一致しません。runtime={feat.shape[0]}, model={expected}\n"
            f"TDoAありモデルを読み込んでいないか確認してください。\n"
            f"音響特徴量のみモデルなら通常は194特徴量です。"
        )

    X = feat.reshape(1, -1)

    pred_label = model.predict(X)[0]

    return label_to_char(pred_label)


# ============================================================
# トリガ音声取得
# ============================================================

def get_trigger_audio(trigger_sample_index: int) -> np.ndarray | None:
    """
    トリガ前後の音声をリングバッファから取得する。
    戻り値は全入力チャンネルの配列。
    """
    pre_samples = int(PRE_TRIGGER_SEC * SAMPLERATE)
    post_samples = int(POST_TRIGGER_SEC * SAMPLERATE)

    start_sample = trigger_sample_index - pre_samples
    end_sample = trigger_sample_index + post_samples

    if start_sample < 0:
        return None

    wait_start = time.perf_counter()

    while audio_buffer.get_total_written() < end_sample:
        if time.perf_counter() - wait_start > AUDIO_WAIT_TIMEOUT_SEC:
            return None

        time.sleep(0.002)

    audio = audio_buffer.read_range(start_sample, end_sample)

    return audio


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    if PREDICT_CHANNEL_INDEX < 0:
        raise ValueError("PREDICT_CHANNEL_INDEX は0以上にしてください。")

    if PREDICT_CHANNEL_INDEX >= INPUT_CHANNELS:
        raise ValueError("PREDICT_CHANNEL_INDEX は INPUT_CHANNELS 未満にしてください。")

    model = load_model(MODEL_PATH)

    log("=== Realtime Typewriter Predictor ===")
    log(f"Model      : {MODEL_PATH}")
    log(f"Serial     : {SERIAL_PORT} / {SERIAL_BAUDRATE}")
    log(f"Device ID  : {DEVICE_ID}")
    log(f"Samplerate : {SAMPLERATE}")
    log(f"Input ch   : {INPUT_CHANNELS}")
    log(f"Predict ch : CH{PREDICT_CHANNEL_INDEX + 1}")
    log("stdoutには推論文字のみを出力します。")

    with sd.InputStream(
        samplerate=SAMPLERATE,
        channels=INPUT_CHANNELS,
        dtype=DTYPE,
        device=DEVICE_ID,
        callback=audio_callback,
    ):
        # リングバッファに少し貯める
        time.sleep(0.5)

        with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=0.05) as ser:
            # Arduino起動待ち
            time.sleep(2.0)
            ser.reset_input_buffer()

            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                if line != "1":
                    # 標準出力を汚さないため、その他のシリアル出力は無視
                    continue

                trigger_sample_index = audio_buffer.get_total_written()
                audio = get_trigger_audio(trigger_sample_index)

                if audio is None:
                    continue

                if audio.ndim != 2:
                    continue

                if PREDICT_CHANNEL_INDEX >= audio.shape[1]:
                    continue

                ch1_audio = audio[:, PREDICT_CHANNEL_INDEX]

                try:
                    char = predict_char(model, ch1_audio)
                except Exception as e:
                    log(f"[ERROR] predict failed: {e}")
                    continue

                # 標準出力には推論文字のみ出す
                print(char, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 標準出力には余計な文字を出さない
        pass