#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "myesp32";
const char* password = "12345678";
const char* udpAddress = "192.168.1.100"; // Python服务器IP
const int udpPort = 12345;

WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200, SERIAL_8N1, 16, 17); // UART1 <-> STM32

  // 连接WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  static uint8_t buffer[50];
  static uint8_t index = 0;

  while (Serial1.available()) {
    uint8_t byte = Serial1.read();
    if (index == 0 && byte != 0xAA) continue;
    if (index == 1 && byte != 0xAF) { index = 0; continue; }

    buffer[index++] = byte;

    if (index == 4) {
      if (buffer[3] > 30) index = 0; // dataLen不合理，丢弃
    }

    if (index > 4 && index == 5 + buffer[3]) {
      uint8_t sum = 0xAA + 0xAF + buffer[2] + buffer[3];
      for (uint8_t i = 0; i < buffer[3]; i++) {
        sum += buffer[4 + i];
      }

      if (sum == buffer[4 + buffer[3]]) {
        // 提取9轴传感器数据
        int16_t accelX = (buffer[4] << 8) | buffer[5];
        int16_t accelY = (buffer[6] << 8) | buffer[7];
        int16_t accelZ = (buffer[8] << 8) | buffer[9];
        int16_t gyroX = (buffer[10] << 8) | buffer[11];
        int16_t gyroY = (buffer[12] << 8) | buffer[13];
        int16_t gyroZ = (buffer[14] << 8) | buffer[15];
        int16_t magX = (buffer[16] << 8) | buffer[17];
        int16_t magY = (buffer[18] << 8) | buffer[19];
        int16_t magZ = (buffer[20] << 8) | buffer[21];
        
        // 准备UDP数据包
        char udpPacket[150];
        snprintf(udpPacket, sizeof(udpPacket), 
                "%d,%d,%d,%d,%d,%d,%d,%d,%d", 
                accelX, accelY, accelZ,
                gyroX, gyroY, gyroZ,
                magX, magY, magZ);
        
        // 发送UDP数据包
        udp.beginPacket(udpAddress, udpPort);
        udp.print(udpPacket);
        udp.endPacket();
        
        Serial.println("数据已发送");
      }
      index = 0;
    }
  }
  
  // 可选: 接收预测结果
  /*
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char incomingPacket[255];
    int len = udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = 0;
    }
    Serial.print("收到预测结果: ");
    Serial.println(incomingPacket);
  }
  */
  
  delay(10); // 适当延迟
}