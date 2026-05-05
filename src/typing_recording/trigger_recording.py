import csv
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import sounddevice as sd
import soundfile as sf


# ============================================================
# 設定
# ============================================================

# ArduinoのCOMポート
SERIAL_PORT = "COM5"

# Arduino側の Serial.begin() と合わせる
SERIAL_BAUDRATE = 1000000

# マイク設定
DEVICE_ID = 12
SAMPLERATE = 96000

# ZOOM AMS-44 ASIO Driver の安定性を考慮し、4chで入力して1ch目だけ保存する
# 本当に1ch入力デバイスを使う場合は INPUT_CHANNELS = 1 に変更する
INPUT_CHANNELS = 4

# 保存するチャンネル番号
# 0: ch1, 1: ch2, 2: ch3, 3: ch4
SAVE_CHANNEL_INDEX = 0

# sounddeviceの録音データ型
DTYPE = "int32"

# トリガ前後の保存時間
PRE_TRIGGER_SEC = 0.015
POST_TRIGGER_SEC = 0.060

# リングバッファ長
# 前100msだけなら短くてもよいが、余裕を見て5秒分保持する
RING_BUFFER_SEC = 5.0

# トリガ後音声待ちタイムアウト
# 音声入力が止まった場合に無限ループしないための時間
AUDIO_WAIT_TIMEOUT_SEC = 1.0

# 保存先
OUTPUT_DIR = Path("recorded_triggers")
OUTPUT_DIR.mkdir(exist_ok=True)

# WAV保存形式
WAV_SUBTYPE = "PCM_24"


# ============================================================
# 音声リングバッファ
# ============================================================

class AudioRingBuffer:
    def __init__(self, samplerate, channels, buffer_sec):
        self.samplerate = samplerate
        self.channels = channels
        self.buffer_samples = int(samplerate * buffer_sec)

        self.buffer = np.zeros(
            (self.buffer_samples, channels),
            dtype=np.int32
        )

        self.write_pos = 0
        self.total_written = 0

        self.lock = threading.Lock()

    def write(self, data):
        """
        data shape: (frames, channels)
        """
        frames = len(data)

        if frames <= 0:
            return

        with self.lock:
            if frames >= self.buffer_samples:
                # バッファ長を超える巨大ブロックが来た場合は末尾のみ保持
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

    def get_total_written(self):
        with self.lock:
            return self.total_written

    def read_range(self, start_abs, end_abs):
        """
        絶対サンプル番号 start_abs 〜 end_abs の範囲を取り出す。
        足りない部分がある場合は None を返す。
        """
        with self.lock:
            current_total = self.total_written
            oldest_available = max(0, current_total - self.buffer_samples)

            if start_abs < oldest_available:
                print("[WARN] 要求範囲の開始位置がリングバッファより古いです。")
                print(f"       start_abs={start_abs}")
                print(f"       oldest_available={oldest_available}")
                return None

            if end_abs > current_total:
                print("[WARN] 要求範囲の終了位置がまだ録音済み範囲に達していません。")
                print(f"       end_abs={end_abs}")
                print(f"       current_total={current_total}")
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
                    self.buffer[:second_len].copy()
                ])

            return audio


audio_buffer = AudioRingBuffer(
    samplerate=SAMPLERATE,
    channels=INPUT_CHANNELS,
    buffer_sec=RING_BUFFER_SEC
)


# ============================================================
# 音声入力コールバック
# ============================================================

def audio_callback(indata, frames, time_info, status):
    try:
        if status:
            print(f"[Audio status] {status}")

        # indataは使い回される可能性があるためcopyする
        audio_buffer.write(indata.copy())

    except Exception as e:
        print("[ERROR] audio_callback内で例外が発生しました")
        print(e)


# ============================================================
# トリガ時の音声保存
# ============================================================

def save_trigger_audio(trigger_count, trigger_sample_index):
    pre_samples = int(PRE_TRIGGER_SEC * SAMPLERATE)
    post_samples = int(POST_TRIGGER_SEC * SAMPLERATE)

    start_sample = trigger_sample_index - pre_samples
    end_sample = trigger_sample_index + post_samples

    print(f"[DEBUG] trigger_sample_index={trigger_sample_index}")
    print(f"[DEBUG] start_sample={start_sample}")
    print(f"[DEBUG] end_sample={end_sample}")
    print(f"[DEBUG] wait until total_written >= {end_sample}")

    if start_sample < 0:
        print("[WARN] 起動直後のため、トリガ前100ms分の音声が不足しています。")
        print("       このトリガは保存しません。")
        return None

    # トリガ後100ms分の音声がリングバッファに入るまで待つ
    # ただし、音声入力が止まった場合に無限待ちしないようタイムアウトする
    wait_start = time.perf_counter()

    while audio_buffer.get_total_written() < end_sample:
        current_total = audio_buffer.get_total_written()

        if time.perf_counter() - wait_start > AUDIO_WAIT_TIMEOUT_SEC:
            print("[ERROR] 音声バッファ待ちでタイムアウトしました。")
            print(f"        current_total={current_total}")
            print(f"        required_total={end_sample}")
            print("        音声入力ストリームが止まっている可能性があります。")
            return None

        time.sleep(0.005)

    print(f"[DEBUG] audio buffer ready. total_written={audio_buffer.get_total_written()}")

    audio = audio_buffer.read_range(start_sample, end_sample)

    if audio is None:
        print("[WARN] トリガ前後の音声を取得できませんでした。")
        print("       起動直後すぎる、またはリングバッファ不足の可能性があります。")
        return None

    # 4chで入力している場合、指定した1chだけ保存する
    if audio.ndim == 2:
        if SAVE_CHANNEL_INDEX >= audio.shape[1]:
            print("[ERROR] SAVE_CHANNEL_INDEX が入力チャンネル数を超えています。")
            print(f"        SAVE_CHANNEL_INDEX={SAVE_CHANNEL_INDEX}")
            print(f"        audio.shape={audio.shape}")
            return None

        audio_to_save = audio[:, SAVE_CHANNEL_INDEX].reshape(-1, 1)
    else:
        audio_to_save = audio.reshape(-1, 1)

    now = datetime.now()
    filename = now.strftime(f"trigger_{trigger_count:04d}_%Y%m%d_%H%M%S_%f.wav")
    filepath = OUTPUT_DIR / filename

    print(f"[DEBUG] writing wav: {filepath}")

    sf.write(
        filepath,
        audio_to_save,
        SAMPLERATE,
        subtype=WAV_SUBTYPE
    )

    print("[DEBUG] wav write done")

    return filepath


# ============================================================
# デバイス情報表示
# ============================================================

def print_audio_device_info():
    print("=== Audio Device Info ===")
    print(f"Device ID   : {DEVICE_ID}")

    device_info = sd.query_devices(DEVICE_ID)
    print(f"Device name : {device_info['name']}")
    print(f"Max input   : {device_info['max_input_channels']}")
    print(f"Max output  : {device_info['max_output_channels']}")
    print(f"Default SR  : {device_info['default_samplerate']}")
    print()


# ============================================================
# メイン処理
# ============================================================

def main():
    if SAVE_CHANNEL_INDEX < 0:
        raise ValueError("SAVE_CHANNEL_INDEX は0以上にしてください。")

    if SAVE_CHANNEL_INDEX >= INPUT_CHANNELS:
        raise ValueError("SAVE_CHANNEL_INDEX は INPUT_CHANNELS 未満にしてください。")

    print("=== Audio Trigger Recorder ===")
    print(f"Serial port : {SERIAL_PORT}")
    print(f"Baudrate    : {SERIAL_BAUDRATE}")
    print(f"Device ID   : {DEVICE_ID}")
    print(f"Device name : {sd.query_devices(DEVICE_ID)['name']}")
    print(f"Samplerate  : {SAMPLERATE}")
    print(f"Input ch    : {INPUT_CHANNELS}")
    print(f"Save ch     : {SAVE_CHANNEL_INDEX + 1}")
    print(f"Save range  : -{PRE_TRIGGER_SEC * 1000:.0f}ms / +{POST_TRIGGER_SEC * 1000:.0f}ms")
    print(f"Output dir  : {OUTPUT_DIR}")
    print()

    print_audio_device_info()

    print("Arduinoから '1' を受信すると、前後100msの音声を保存します。")
    print("Ctrl + C で終了します。")
    print()

    log_path = OUTPUT_DIR / "trigger_log.csv"

    trigger_count = 0

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "trigger_no",
            "timestamp_iso",
            "trigger_sample_index",
            "wav_file"
        ])

        # 音声入力開始
        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=INPUT_CHANNELS,
            dtype=DTYPE,
            device=DEVICE_ID,
            callback=audio_callback
        ):
            print("[INFO] 音声入力を開始しました。")
            print("[INFO] リングバッファ準備のため、少し待機します。")
            time.sleep(0.5)

            print(f"[DEBUG] audio total_written after warmup={audio_buffer.get_total_written()}")

            # シリアル接続
            with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1) as ser:
                print("[INFO] シリアルポートを開きました。")
                print("[INFO] Arduino起動待ち中...")
                time.sleep(2.0)

                # 起動直後のゴミデータを捨てる
                ser.reset_input_buffer()

                print("[INFO] トリガ待機中...")

                while True:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()

                    if not line:
                        continue

                    if line == "1":
                        trigger_count += 1

                        now = datetime.now()

                        # 受信時点における音声サンプル位置をトリガ位置とする
                        trigger_sample_index = audio_buffer.get_total_written()

                        print(f"[TRIGGER] {trigger_count}: {now.isoformat(timespec='milliseconds')}")
                        print(f"[DEBUG] audio total_written={trigger_sample_index}")

                        filepath = save_trigger_audio(
                            trigger_count,
                            trigger_sample_index
                        )

                        if filepath is not None:
                            print(f"[SAVE] {filepath}")

                            log_writer.writerow([
                                trigger_count,
                                now.isoformat(timespec="milliseconds"),
                                trigger_sample_index,
                                filepath.name
                            ])
                            log_file.flush()
                        else:
                            print("[WARN] WAV保存に失敗、またはスキップしました。")

                    else:
                        print(f"[Serial] {line}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了しました。")