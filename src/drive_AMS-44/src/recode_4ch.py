import sounddevice as sd
import soundfile as sf
import numpy as np

samplerate = 96000
channels = 4
duration = 5
device_id = 18  # ここでマイクのIDを指定

print(f"Using device #{device_id}: {sd.query_devices(device_id)['name']}")

# 録音
recorded = sd.rec(int(duration * samplerate),
                  samplerate=samplerate,
                  channels=channels,
                  dtype='int32',
                  device=device_id)
sd.wait()

# --- ★追加：4chを1ファイルで保存 ---
sf.write('all_channels.wav',
         recorded,
         samplerate,
         subtype='PCM_24')
print("Saved all_channels.wav (4ch interleaved)")

# チャンネル分割保存（従来）
for ch in range(channels):
    sf.write(f'channel_{ch+1}.wav',
             recorded[:, ch].reshape(-1, 1),
             samplerate,
             subtype='PCM_24')
    print(f"Saved channel_{ch+1}.wav")