//==============================================================
// AE-BMX055 9軸センサーモジュール
// Arduino Pro Micro 用
//
// 用途：タイプライターのキー押下検出
//
// 取得軸　：加速度 Z軸のみ
// 取得周期：1kHz
//
// 押下検出条件：
//   |Z - baseline| > 0.5g
//
// 押下検出後：
//   300ms間は再検出しない
//
// シリアル送信：
//   押下検出時のみ "1" を送信
//==============================================================

#include <Wire.h>

// BMX055 加速度センサ I2Cアドレス
#define Addr_Accl 0x19

// サンプリング周期
// 1kHz = 1000us
const unsigned long SAMPLE_PERIOD_US = 1000;

// 押下検出しきい値[g]
const float TRIGGER_THRESHOLD_G = 0.5;

// 押下検出後の無視時間[ms]
const unsigned long IGNORE_PERIOD_MS = 600;

// 起動時の基準値取得サンプル数
// 1kHzなので500サンプル = 約0.5秒
const int CALIBRATION_SAMPLE_COUNT = 500;

// Z軸加速度[g]
float zAccl = 0.0;

// Z軸基準値[g]
float zBaseline = 0.0;

// 基準値取得用
bool calibrated = false;
int calibrationCount = 0;
float calibrationSum = 0.0;

// 時間管理
unsigned long nextSampleTime = 0;
unsigned long lastTriggerTimeMs = 0;

void setup()
{
  Wire.begin();

  // I2Cを400kHzに高速化
  Wire.setClock(400000);

  Serial.begin(1000000);

  // Pro MicroのUSBシリアル待ち
  // 単体動作時に止まらないよう最大5秒
  unsigned long startTime = millis();
  while (!Serial && millis() - startTime < 5000) {
    ;
  }

  BMX055_Accl_Init();

  delay(100);

  nextSampleTime = micros();
}

void loop()
{
  unsigned long now = micros();

  //============================================================
  // 1kHzでZ軸加速度を取得
  //============================================================
  if ((long)(now - nextSampleTime) >= 0) {
    nextSampleTime += SAMPLE_PERIOD_US;

    bool readOk = BMX055_Accl_Read_Z();

    if (!readOk) {
      return;
    }

    //==========================================================
    // 起動直後にZ軸の基準値を取得
    // この間は押下検出しない
    //==========================================================
    if (!calibrated) {
      calibrationSum += zAccl;
      calibrationCount++;

      if (calibrationCount >= CALIBRATION_SAMPLE_COUNT) {
        zBaseline = calibrationSum / calibrationCount;
        calibrated = true;
      }

      return;
    }

    //==========================================================
    // 押下検出
    //==========================================================
    float diff = zAccl - zBaseline;
    if (diff < 0) {
      diff = -diff;
    }

    unsigned long nowMs = millis();

    if (diff > TRIGGER_THRESHOLD_G) {
      // 前回検出から300ms以上経過していれば押下と判定
      if (nowMs - lastTriggerTimeMs >= IGNORE_PERIOD_MS) {
        Serial.println(1);

        lastTriggerTimeMs = nowMs;
      }
    }
  }
}

//==============================================================
// BMX055 加速度センサ初期化
//==============================================================
void BMX055_Accl_Init()
{
  // 加速度レンジ設定
  // 0x03 = ±2g
  Wire.beginTransmission(Addr_Accl);
  Wire.write(0x0F);
  Wire.write(0x03);
  Wire.endTransmission();
  delay(10);

  // 帯域設定
  // 0x0F = 1000Hz
  Wire.beginTransmission(Addr_Accl);
  Wire.write(0x10);
  Wire.write(0x0F);
  Wire.endTransmission();
  delay(10);

  // 通常動作モード
  Wire.beginTransmission(Addr_Accl);
  Wire.write(0x11);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(10);
}

//==============================================================
// BMX055 加速度 Z軸のみ読み取り
//==============================================================
bool BMX055_Accl_Read_Z()
{
  uint8_t zLsb;
  uint8_t zMsb;

  // Z軸データレジスタ
  // 0x06: Z軸 LSB
  // 0x07: Z軸 MSB
  Wire.beginTransmission(Addr_Accl);
  Wire.write(0x06);
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)Addr_Accl, (uint8_t)2);

  if (Wire.available() < 2) {
    return false;
  }

  zLsb = Wire.read();
  zMsb = Wire.read();

  // 12bit符号付き値に変換
  int16_t rawZ = (((int16_t)zMsb << 8) | (zLsb & 0xF0)) >> 4;

  // 符号拡張
  if (rawZ > 2047) {
    rawZ -= 4096;
  }

  // ±2g設定時の換算
  zAccl = rawZ * 0.00098;

  return true;
}