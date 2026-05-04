//==============================================================
// AE-BMX055 9軸センサーモジュール
// Arduino Pro Micro 用
//
// 取得軸　：加速度 Z軸のみ
// 取得周期：1kHz
// 送信周期：1kHz
// 送信内容：Z軸加速度[g]のみ
//==============================================================

#include <Wire.h>

// BMX055 加速度センサ I2Cアドレス
#define Addr_Accl 0x19

// サンプリング周期
// 1kHz = 1000us
const unsigned long SAMPLE_PERIOD_US = 1000;

// シリアル送信周期
// 解析用に1kHzで送信する
// 10Hzにしたい場合は 100000 に変更
const unsigned long SERIAL_PERIOD_US = 1000;

// Z軸加速度[g]
float zAccl = 0.0;

unsigned long nextSampleTime = 0;
unsigned long nextSerialTime = 0;

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

  unsigned long now = micros();
  nextSampleTime = now;
  nextSerialTime = now;
}

void loop()
{
  unsigned long now = micros();

  //============================================================
  // 1kHzでZ軸加速度を取得
  //============================================================
  if ((long)(now - nextSampleTime) >= 0) {
    nextSampleTime += SAMPLE_PERIOD_US;

    BMX055_Accl_Read_Z();
  }

  //============================================================
  // 1kHzでZ軸のみシリアル送信
  //============================================================
  if ((long)(now - nextSerialTime) >= 0) {
    nextSerialTime += SERIAL_PERIOD_US;

    // CSV保存しやすいように、Z軸の数値のみ送信
    Serial.println(zAccl, 4);
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
void BMX055_Accl_Read_Z()
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
    return;
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
}