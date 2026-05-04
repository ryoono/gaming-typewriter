import serial
import csv
import time
from datetime import datetime
from pathlib import Path

# ==============================
# 設定
# ==============================

# ArduinoのCOMポートに合わせて変更してください
PORT = "COM3"

# Arduino側の Serial.begin() と合わせる
BAUDRATE = 1000000

# 保存先フォルダ
OUTPUT_DIR = Path(".")

# CSVファイル名
filename = datetime.now().strftime("bmx055_z_log_%Y%m%d_%H%M%S.csv")
filepath = OUTPUT_DIR / filename

# ==============================
# 受信・保存処理
# ==============================

def main():
    print(f"接続ポート: {PORT}")
    print(f"ボーレート: {BAUDRATE}")
    print(f"保存ファイル: {filepath}")
    print("Ctrl + C で終了します。")

    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        # Arduinoのリセット・起動待ち
        time.sleep(2)

        # 起動直後のゴミデータを捨てる
        ser.reset_input_buffer()

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # CSVヘッダ
            writer.writerow([
                "timestamp_iso",
                "elapsed_sec",
                "z_g"
            ])

            start_time = time.perf_counter()

            try:
                while True:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()

                    if not line:
                        continue

                    try:
                        z_g = float(line)
                    except ValueError:
                        print(f"skip: {line}")
                        continue

                    now = datetime.now()
                    elapsed_sec = time.perf_counter() - start_time

                    writer.writerow([
                        now.isoformat(timespec="milliseconds"),
                        f"{elapsed_sec:.6f}",
                        f"{z_g:.4f}"
                    ])

                    # 逐次保存したい場合
                    f.flush()

                    print(f"{now.isoformat(timespec='milliseconds')}, {elapsed_sec:.6f}, {z_g:.4f}")

            except KeyboardInterrupt:
                print("\n保存を終了しました。")
                print(f"CSVファイル: {filepath}")

if __name__ == "__main__":
    main()