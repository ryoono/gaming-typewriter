# batch_typewriter_tdoa.py
# -*- coding: utf-8 -*-

"""
タイプライター 2ch WAV 一括TDoA解析プログラム

想定:
- CH1 = 左マイク
- CH2 = 右マイク
- マイクはタイプライター左右に設置
- 2つ目の山にアーム戻り音が含まれる
- 2つ目イベントの後半側を使って左右時間差を推定する

出力:
- results.csv
- 各WAVごとの確認用PNG

使い方:
    python batch_typewriter_tdoa.py ./wav_folder

必要ライブラリ:
    pip install numpy scipy pandas matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


@dataclass
class TdoaConfig:
    mic_distance_m: float = 0.21
    sound_speed_m_s: float = 343.0

    # 2つ目イベントを探し始める時刻
    second_search_start_ms: float = 120.0

    # イベント検出用帯域
    event_band_hz: tuple[float, float] = (300.0, 20000.0)

    # 2つ目イベント後半の「戻り衝撃」っぽいピーク検出用帯域
    high_band_hz: tuple[float, float] = (3000.0, 20000.0)

    # TDoA計算用帯域
    # 高すぎると波形の周期・位相差に引っ張られやすく、
    # 低すぎると筐体振動・キャリッジ音が混ざりやすいので、
    # まずは500〜4000Hzを使う
    tdoa_band_hz: tuple[float, float] = (500.0, 4000.0)

    # 後半ピークの直前を使う
    tdoa_pre_ms: float = 3.0
    tdoa_post_ms: float = 0.2

    # 2つ目イベント領域の中で「後半の有意な高域ピーク」を選ぶための比率
    # 高域包絡線の最大値に対して、この比率以上のピークのうち最後のものを使う
    tail_peak_ratio: float = 0.45

    # 包絡線の平滑化
    envelope_smooth_ms: float = 0.20

    # イベントピーク検出の最小間隔
    min_event_distance_ms: float = 50.0


def pcm_to_float(x: np.ndarray) -> np.ndarray:
    """PCM整数をfloat [-1, 1] 付近に変換"""
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), abs(info.max))
        return x.astype(np.float64) / scale
    return x.astype(np.float64)


def bandpass(x: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)

    if low >= high:
        raise ValueError(f"Invalid band: low={low_hz}, high={high_hz}, fs={fs}")

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def smooth_moving_average(x: np.ndarray, fs: int, smooth_ms: float) -> np.ndarray:
    n = max(1, int(round(fs * smooth_ms / 1000.0)))
    if n <= 1:
        return x
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def envelope_band(x: np.ndarray, fs: int, band_hz: tuple[float, float], smooth_ms: float) -> np.ndarray:
    y = bandpass(x, fs, band_hz[0], band_hz[1])
    env = np.abs(hilbert(y))
    env = smooth_moving_average(env, fs, smooth_ms)
    return env


def normalize_for_corr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return x * 0.0
    return x / std


def corr_at_lag(ch1: np.ndarray, ch2: np.ndarray, lag: int) -> float:
    """
    lagの定義:
        lag = t_CH2 - t_CH1 のサンプル数

    lag > 0:
        CH2がCH1より遅い、つまりCH1が先に鳴った
    lag < 0:
        CH2がCH1より早い
    """
    if lag > 0:
        a = ch1[:-lag]
        b = ch2[lag:]
    elif lag < 0:
        a = ch1[-lag:]
        b = ch2[:lag]
    else:
        a = ch1
        b = ch2

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
    fs: int,
    max_lag_s: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    相互相関でTDoAを推定する。

    戻り値:
        lag_samples_float:
            t_CH2 - t_CH1 のサンプル数
        best_corr:
            最大相関値
        lags:
            探索した整数lag
        corrs:
            各lagの相関値
    """
    max_lag_samples = int(round(max_lag_s * fs))
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)

    corrs = np.array([corr_at_lag(ch1_feature, ch2_feature, int(lag)) for lag in lags])

    if np.all(np.isnan(corrs)):
        return np.nan, np.nan, lags, corrs

    best_i = int(np.nanargmax(corrs))
    best_lag = float(lags[best_i])
    best_corr = float(corrs[best_i])

    # サブサンプル推定: 相関ピーク周辺3点で放物線補間
    if 0 < best_i < len(corrs) - 1:
        y0 = corrs[best_i - 1]
        y1 = corrs[best_i]
        y2 = corrs[best_i + 1]

        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12 and not np.isnan(denom):
            offset = 0.5 * (y0 - y2) / denom
            offset = float(np.clip(offset, -1.0, 1.0))
            best_lag += offset

    return best_lag, best_corr, lags, corrs


def find_second_event_peak(event_env: np.ndarray, fs: int, cfg: TdoaConfig) -> int:
    """
    大まかな2つ目イベント位置を探す。
    まずピーク検出を試し、難しければsecond_search_start以降の最大値を使う。
    """
    distance = int(round(fs * cfg.min_event_distance_ms / 1000.0))
    height = np.max(event_env) * 0.12
    prominence = np.max(event_env) * 0.06

    peaks, _ = find_peaks(event_env, distance=distance, height=height, prominence=prominence)

    second_start = int(round(fs * cfg.second_search_start_ms / 1000.0))

    # second_search_start以降のピークを優先
    peaks_after = peaks[peaks >= second_start]
    if len(peaks_after) > 0:
        return int(peaks_after[0])

    # ピーク検出できない場合は、second_search_start以降の最大値
    if second_start < len(event_env):
        return int(second_start + np.argmax(event_env[second_start:]))

    # 短すぎる場合は全体の最大値
    return int(np.argmax(event_env))


def find_event_region(env: np.ndarray, peak_idx: int, fs: int, threshold_ratio: float = 0.12) -> tuple[int, int]:
    """
    指定ピークの周辺で、包絡線がしきい値以上の領域をざっくり取得。
    """
    peak_val = env[peak_idx]
    th = peak_val * threshold_ratio

    left = peak_idx
    while left > 1 and env[left] > th:
        left -= 1

    right = peak_idx
    while right < len(env) - 2 and env[right] > th:
        right += 1

    # 領域が狭すぎる場合の保険
    min_half = int(round(fs * 8.0 / 1000.0))
    if right - left < min_half:
        left = max(0, peak_idx - min_half)
        right = min(len(env), peak_idx + min_half)

    return left, right


def find_tail_high_peak(
    hf_env_mix: np.ndarray,
    region_start: int,
    region_end: int,
    fs: int,
    cfg: TdoaConfig,
) -> int:
    """
    2つ目イベント領域の中から、後半側の高域ピークを選ぶ。

    方針:
    - 領域内の高域包絡線ピークを取る
    - 最大ピークのtail_peak_ratio以上の高さを持つピークのうち、
      時間的に最後のものを採用する
    - これにより、立ち上がり側のキャリッジ音ではなく、
      終わり側のアーム戻り音を拾いやすくする
    """
    seg = hf_env_mix[region_start:region_end]
    if len(seg) < 8:
        return int(region_start + np.argmax(seg))

    max_val = float(np.max(seg))
    if max_val <= 0:
        return int(region_start + np.argmax(seg))

    # 高域ピーク同士が近すぎると細かい振動を拾うので、0.4ms程度は離す
    distance = max(1, int(round(fs * 0.4 / 1000.0)))
    peaks, props = find_peaks(seg, height=max_val * cfg.tail_peak_ratio, distance=distance)

    if len(peaks) > 0:
        # 条件を満たすピークのうち最後
        return int(region_start + peaks[-1])

    # 見つからなければ、領域内の後半半分の最大値
    mid = len(seg) // 2
    return int(region_start + mid + np.argmax(seg[mid:]))


def analyze_wav(path: Path, output_dir: Path, cfg: TdoaConfig, make_plot: bool = True) -> dict:
    fs, data = wavfile.read(path)
    data = pcm_to_float(data)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path.name}: 2ch WAVではありません")

    ch1 = data[:, 0]
    ch2 = data[:, 1]
    n = len(ch1)
    t = np.arange(n) / fs

    # 物理的にあり得る最大遅延
    max_lag_s = cfg.mic_distance_m / cfg.sound_speed_m_s

    # イベント検出用包絡線
    env1_event = envelope_band(ch1, fs, cfg.event_band_hz, cfg.envelope_smooth_ms)
    env2_event = envelope_band(ch2, fs, cfg.event_band_hz, cfg.envelope_smooth_ms)
    event_env = 0.5 * (env1_event + env2_event)

    second_peak = find_second_event_peak(event_env, fs, cfg)
    region_start, region_end = find_event_region(event_env, second_peak, fs)

    # 高域包絡線で後半側ピークを探す
    env1_hf = envelope_band(ch1, fs, cfg.high_band_hz, cfg.envelope_smooth_ms)
    env2_hf = envelope_band(ch2, fs, cfg.high_band_hz, cfg.envelope_smooth_ms)
    hf_env_mix = 0.5 * (env1_hf + env2_hf)

    tail_peak = find_tail_high_peak(hf_env_mix, region_start, region_end, fs, cfg)

    # TDoA計算用包絡線
    env1_tdoa = envelope_band(ch1, fs, cfg.tdoa_band_hz, cfg.envelope_smooth_ms)
    env2_tdoa = envelope_band(ch2, fs, cfg.tdoa_band_hz, cfg.envelope_smooth_ms)

    pre = int(round(fs * cfg.tdoa_pre_ms / 1000.0))
    post = int(round(fs * cfg.tdoa_post_ms / 1000.0))

    win_start = max(0, tail_peak - pre)
    win_end = min(n, tail_peak + post)

    f1 = env1_tdoa[win_start:win_end]
    f2 = env2_tdoa[win_start:win_end]

    lag_samples, best_corr, lags, corrs = estimate_lag_by_xcorr(
        f1, f2, fs=fs, max_lag_s=max_lag_s
    )

    delta_ms = lag_samples / fs * 1000.0 if not np.isnan(lag_samples) else np.nan

    # 補助特徴量: 左右エネルギー比
    eps = 1e-12
    rms1 = float(np.sqrt(np.mean(f1**2) + eps))
    rms2 = float(np.sqrt(np.mean(f2**2) + eps))
    energy_lr_db = 20.0 * np.log10((rms1 + eps) / (rms2 + eps))

    # 1回目イベントでゼロ点チェック
    first_delta_ms = np.nan
    corrected_delta_ms = np.nan

    first_search_end = int(round(fs * cfg.second_search_start_ms / 1000.0))
    if first_search_end > int(10e-3 * fs):
        first_peak = int(np.argmax(event_env[:first_search_end]))
        first_region_start, first_region_end = find_event_region(event_env, first_peak, fs)
        first_tail_peak = find_tail_high_peak(hf_env_mix, first_region_start, first_region_end, fs, cfg)

        fw_start = max(0, first_tail_peak - pre)
        fw_end = min(n, first_tail_peak + post)

        ff1 = env1_tdoa[fw_start:fw_end]
        ff2 = env2_tdoa[fw_start:fw_end]

        first_lag_samples, first_corr, _, _ = estimate_lag_by_xcorr(
            ff1, ff2, fs=fs, max_lag_s=max_lag_s
        )

        if not np.isnan(first_lag_samples):
            first_delta_ms = first_lag_samples / fs * 1000.0
            corrected_delta_ms = delta_ms - first_delta_ms

    result = {
        "file": path.name,
        "fs_Hz": fs,
        "duration_s": n / fs,
        "second_event_peak_ms": second_peak / fs * 1000.0,
        "second_region_start_ms": region_start / fs * 1000.0,
        "second_region_end_ms": region_end / fs * 1000.0,
        "tail_peak_ms": tail_peak / fs * 1000.0,
        "tdoa_window_start_ms": win_start / fs * 1000.0,
        "tdoa_window_end_ms": win_end / fs * 1000.0,
        "tdoa_raw_ms_CH2_minus_CH1": delta_ms,
        "first_event_tdoa_ms_CH2_minus_CH1": first_delta_ms,
        "tdoa_corrected_ms": corrected_delta_ms,
        "best_corr": best_corr,
        "energy_CH1_over_CH2_dB": energy_lr_db,
        "judgement_raw": judge_direction(delta_ms),
        "judgement_corrected": judge_direction(corrected_delta_ms),
    }

    if make_plot:
        save_debug_plot(
            path=path,
            output_dir=output_dir,
            fs=fs,
            t=t,
            ch1=ch1,
            ch2=ch2,
            event_env=event_env,
            hf_env_mix=hf_env_mix,
            env1_tdoa=env1_tdoa,
            env2_tdoa=env2_tdoa,
            region_start=region_start,
            region_end=region_end,
            second_peak=second_peak,
            tail_peak=tail_peak,
            win_start=win_start,
            win_end=win_end,
            lags=lags,
            corrs=corrs,
            result=result,
        )

    return result


def judge_direction(delta_ms: float, dead_zone_ms: float = 0.03) -> str:
    """
    delta = t_CH2 - t_CH1

    delta > 0:
        CH1が先、左寄り
    delta < 0:
        CH2が先、右寄り
    """
    if delta_ms is None or np.isnan(delta_ms):
        return "unknown"
    if delta_ms > dead_zone_ms:
        return "left_CH1_early"
    if delta_ms < -dead_zone_ms:
        return "right_CH2_early"
    return "near_center"


def save_debug_plot(
    path: Path,
    output_dir: Path,
    fs: int,
    t: np.ndarray,
    ch1: np.ndarray,
    ch2: np.ndarray,
    event_env: np.ndarray,
    hf_env_mix: np.ndarray,
    env1_tdoa: np.ndarray,
    env2_tdoa: np.ndarray,
    region_start: int,
    region_end: int,
    second_peak: int,
    tail_peak: int,
    win_start: int,
    win_end: int,
    lags: np.ndarray,
    corrs: np.ndarray,
    result: dict,
) -> None:
    png_path = output_dir / f"{path.stem}_tdoa_debug.png"

    # 表示範囲: 2つ目イベント周辺
    margin_ms = 35.0
    x0 = max(0, region_start - int(round(fs * margin_ms / 1000.0)))
    x1 = min(len(ch1), region_end + int(round(fs * margin_ms / 1000.0)))

    tt_ms = t[x0:x1] * 1000.0

    # 見やすいように正規化
    def norm(x: np.ndarray) -> np.ndarray:
        m = np.max(np.abs(x))
        if m < 1e-12:
            return x
        return x / m

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(tt_ms, norm(ch1[x0:x1]), label="CH1 raw left", alpha=0.8)
    ax1.plot(tt_ms, norm(ch2[x0:x1]), label="CH2 raw right", alpha=0.8)
    ax1.axvline(second_peak / fs * 1000.0, linestyle="--", label="second event peak")
    ax1.axvline(tail_peak / fs * 1000.0, linestyle=":", label="tail HF peak")
    ax1.axvspan(win_start / fs * 1000.0, win_end / fs * 1000.0, alpha=0.2, label="TDoA window")
    ax1.set_title(path.name)
    ax1.set_ylabel("raw normalized")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(tt_ms, norm(event_env[x0:x1]), label="event envelope 300-20000Hz")
    ax2.plot(tt_ms, norm(hf_env_mix[x0:x1]), label="HF envelope 3000-20000Hz")
    ax2.plot(tt_ms, norm(env1_tdoa[x0:x1]), label="CH1 TDoA env 500-4000Hz", alpha=0.8)
    ax2.plot(tt_ms, norm(env2_tdoa[x0:x1]), label="CH2 TDoA env 500-4000Hz", alpha=0.8)
    ax2.axvspan(region_start / fs * 1000.0, region_end / fs * 1000.0, alpha=0.12, label="second region")
    ax2.axvspan(win_start / fs * 1000.0, win_end / fs * 1000.0, alpha=0.2, label="TDoA window")
    ax2.set_ylabel("envelope normalized")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(3, 1, 3)
    if lags is not None and corrs is not None and len(lags) == len(corrs):
        lag_ms = lags / fs * 1000.0
        ax3.plot(lag_ms, corrs)
        ax3.axvline(result["tdoa_raw_ms_CH2_minus_CH1"], linestyle="--", label="estimated TDoA")
    ax3.set_xlabel("lag = t_CH2 - t_CH1 [ms]")
    ax3.set_ylabel("normalized correlation")
    ax3.grid(True)
    ax3.legend(loc="upper right")

    raw = result["tdoa_raw_ms_CH2_minus_CH1"]
    corr = result["tdoa_corrected_ms"]
    fig.suptitle(
        f"raw Δt={raw:.4f} ms, corrected Δt={corr:.4f} ms, "
        f"judge={result['judgement_corrected']}, corr={result['best_corr']:.3f}",
        y=0.98,
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TDoA analysis for 2ch typewriter WAV files")
    parser.add_argument("input_dir", type=str, help="WAVファイルが入ったフォルダ")
    parser.add_argument("--output-dir", type=str, default="tdoa_output", help="出力フォルダ")
    parser.add_argument("--pattern", type=str, default="*.wav", help="WAV検索パターン")
    parser.add_argument("--mic-distance", type=float, default=0.21, help="マイク間距離[m]")
    parser.add_argument("--sound-speed", type=float, default=343.0, help="音速[m/s]")
    parser.add_argument("--second-search-start-ms", type=float, default=120.0, help="2つ目イベント探索開始時刻[ms]")
    parser.add_argument("--no-plot", action="store_true", help="PNGを出力しない")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TdoaConfig(
        mic_distance_m=args.mic_distance,
        sound_speed_m_s=args.sound_speed,
        second_search_start_ms=args.second_search_start_ms,
    )

    wavs = sorted(input_dir.glob(args.pattern))
    if not wavs:
        print(f"WAVが見つかりません: {input_dir / args.pattern}")
        return

    results = []

    for wav in wavs:
        print(f"processing: {wav.name}")
        try:
            r = analyze_wav(wav, output_dir, cfg, make_plot=not args.no_plot)
            results.append(r)

            print(
                f"  Δt raw = {r['tdoa_raw_ms_CH2_minus_CH1']:+.4f} ms, "
                f"corrected = {r['tdoa_corrected_ms']:+.4f} ms, "
                f"judge = {r['judgement_corrected']}, "
                f"corr = {r['best_corr']:.3f}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "file": wav.name,
                "error": str(e),
            })

    df = pd.DataFrame(results)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print()
    print(f"done: {csv_path}")


if __name__ == "__main__":
    main()