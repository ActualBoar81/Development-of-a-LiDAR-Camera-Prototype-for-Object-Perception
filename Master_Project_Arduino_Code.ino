#include <Wire.h>
#include <Servo.h>

const uint8_t TFMINI_ADDR = 0x10;

Servo servoYaw;
Servo servoPitch;

int yawPin = 9;
int pitchPin = 2;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  servoYaw.attach(yawPin);
  servoPitch.attach(pitchPin);

  delay(300);
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("ANGLE")) {
      int firstComma = command.indexOf(',');
      int secondComma = command.indexOf(',', firstComma + 1);

      if (firstComma == -1 || secondComma == -1) {
        Serial.println("ERROR,BAD_COMMAND");
        return;
      }

      int yawAngle = command.substring(firstComma + 1, secondComma).toInt();
      int pitchAngle = command.substring(secondComma + 1).toInt();

      yawAngle = constrain(yawAngle, 0, 180);
      pitchAngle = constrain(pitchAngle, 0, 180);

      servoYaw.write(yawAngle);
      servoPitch.write(pitchAngle);

      delay(500);

      int distance = readTFminiDistance();

      if (distance >= 0) {
        Serial.print("DIST,");
        Serial.println(distance);
      } else {
        Serial.println("ERROR,TFMINI");
      }
    } else {
      Serial.println("ERROR,UNKNOWN_COMMAND");
    }
  }
}

int readTFminiDistance() {
  uint8_t rx_buf[9] = {0};
  uint8_t checksum = 0;
  uint8_t count = 0;

  Wire.beginTransmission(TFMINI_ADDR);
  Wire.write(0x5A);
  Wire.write(0x05);
  Wire.write(0x00);
  Wire.write(0x01);
  Wire.write(0x60);

  uint8_t err = Wire.endTransmission();

  if (err != 0) {
    return -1;
  }

  delay(10);

  Wire.requestFrom(TFMINI_ADDR, (uint8_t)9);

  while (Wire.available() && count < 9) {
    rx_buf[count++] = Wire.read();
  }

  if (count != 9) {
    return -1;
  }

  for (uint8_t i = 0; i < 8; i++) {
    checksum += rx_buf[i];
  }

  if (rx_buf[0] == 0x59 && rx_buf[1] == 0x59 && rx_buf[8] == checksum) {
    uint16_t distance = rx_buf[2] | (rx_buf[3] << 8);
    return distance;
  }

  return -1;
}
