//==============================================================
// AE-BMX055 9軸センサーモジュール
// Arduino Pro Micro 用
//
// 取得周期：1kHz
// 表示周期：10Hz
// 表示内容：加速度 X, Y, Z のみ
//==============================================================

#include <Wire.h>

// BMX055 加速度センサ I2Cアドレス
#define Addr_Accl 0x19

// サンプリング周期
// 1kHz = 1000us
const unsigned long SAMPLE_PERIOD_US = 1000;

// シリアル表示周期
// 10Hz = 100ms = 100000us
const unsigned long SERIAL_PERIOD_US = 100000;

// 加速度[g]
float xAccl = 0.0;
float yAccl = 0.0;
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

  Serial.println("X[g],Y[g],Z[g]");

  unsigned long now = micros();
  nextSampleTime = now;
  nextSerialTime = now;
}

void loop()
{
  unsigned long now = micros();

  //============================================================
  // 1kHzで加速度を取得
  //============================================================
  if ((long)(now - nextSampleTime) >= 0) {
    nextSampleTime += SAMPLE_PERIOD_US;

    BMX055_Accl_Read();
  }

  //============================================================
  // 10Hzでシリアルモニタへ表示
  //============================================================
  if ((long)(now - nextSerialTime) >= 0) {
    nextSerialTime += SERIAL_PERIOD_US;

    Serial.print(xAccl, 4);
    Serial.print(",");
    Serial.print(yAccl, 4);
    Serial.print(",");
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
// BMX055 加速度読み取り
//==============================================================
void BMX055_Accl_Read()
{
  uint8_t data[6];

  // 加速度データレジスタ 0x02 から6バイト読み出し
  Wire.beginTransmission(Addr_Accl);
  Wire.write(0x02);
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)Addr_Accl, (uint8_t)6);

  if (Wire.available() < 6) {
    return;
  }

  for (int i = 0; i < 6; i++) {
    data[i] = Wire.read();
  }

  // 12bit符号付き値に変換
  int16_t rawX = ((int16_t)data[1] << 8 | (data[0] & 0xF0)) >> 4;
  int16_t rawY = ((int16_t)data[3] << 8 | (data[2] & 0xF0)) >> 4;
  int16_t rawZ = ((int16_t)data[5] << 8 | (data[4] & 0xF0)) >> 4;

  // 符号拡張
  if (rawX > 2047) rawX -= 4096;
  if (rawY > 2047) rawY -= 4096;
  if (rawZ > 2047) rawZ -= 4096;

  // ±2g設定時の換算
  xAccl = rawX * 0.00098;
  yAccl = rawY * 0.00098;
  zAccl = rawZ * 0.00098;
}
