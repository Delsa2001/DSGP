#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// -- WiFi & Cloud -------------------------------------------------------------
const char* WIFI_SSID     = "SLT-4G-7F34";
const char* WIFI_PASSWORD = "A823RQ015BT";
const char* SERVER_URL    = "https://edgeaitool.vercel.app/data";
const char* COMMAND_URL   = "https://edgeaitool.vercel.app/command";

// -- Pin Definitions ----------------------------------------------------------
#define DS18B20_PIN   5
#define MOISTURE_PIN  4
#define MQ4_AO_PIN    3
#define MQ4_DO_PIN    6
#define OLED_SDA      8
#define OLED_SCL      9

// -- OLED ---------------------------------------------------------------------
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// -- DS18B20 ------------------------------------------------------------------
OneWire oneWire(DS18B20_PIN);
DallasTemperature tempSensor(&oneWire);

// -- Rule-Based Classifier Thresholds ----------------------------------------
#define METHANE_THRESH    1341
#define TEMP_THRESH       29.09f
#define MOISTURE_THRESH   1906

// -- Timing -------------------------------------------------------------------
unsigned long lastSendTime     = 0;
const long    SEND_INTERVAL    = 10000;
unsigned long lastCommandTime  = 0;
const long    COMMAND_INTERVAL = 15000;

void setup() {
  Serial.begin(115200);

  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED init failed");
  }
  showMessage("Booting...", "", "");

  tempSensor.begin();
  pinMode(MQ4_DO_PIN, INPUT);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  showMessage("Connecting", "WiFi...", "");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected: " + WiFi.localIP().toString());
    showMessage("WiFi OK", WiFi.localIP().toString().c_str(), "");
  } else {
    showMessage("WiFi FAILED", "Offline mode", "");
  }
  delay(1500);
}

void loop() {
  float temperature = readTemperature();
  float moisture    = readMoisture();
  float methane     = readMethane();

  String result = classify(temperature, (int)moisture, (int)methane);

  showReading(temperature, moisture, methane, result);
  Serial.printf("Result: %s\n", result.c_str());

  if (millis() - lastSendTime > SEND_INTERVAL) {
    sendToCloud(temperature, moisture, methane, result);
    lastSendTime = millis();
  }

  if (millis() - lastCommandTime > COMMAND_INTERVAL) {
    checkCommand();
    lastCommandTime = millis();
  }

  delay(2000);
}

// Moisture-priority classification to match updated cloud logic.
String classify(float temp, int moisture, int methane) {
  if (moisture <= MOISTURE_THRESH)  return "TOO_WET";
  if (temp > TEMP_THRESH)           return "TOO_DRY";
  if (methane <= METHANE_THRESH)    return "COMPOST_READY";
  return "TOO_DRY";
}

float readTemperature() {
  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  if (t == DEVICE_DISCONNECTED_C) {
    Serial.println("DS18B20 error");
    return 0.0;
  }
  Serial.printf("Temp: %.2f C\n", t);
  return t;
}

float readMoisture() {
  long sum = 0;
  for (int i = 0; i < 5; i++) { sum += analogRead(MOISTURE_PIN); delay(10); }
  int raw = (int)(sum / 5);
  Serial.printf("Moisture ADC: %d\n", raw);
  return (float)raw;
}

float readMethane() {
  long sum = 0;
  for (int i = 0; i < 5; i++) { sum += analogRead(MQ4_AO_PIN); delay(10); }
  int raw = (int)(sum / 5);
  Serial.printf("Methane ADC: %d\n", raw);
  return (float)raw;
}

void showReading(float temp, float moisture, float methane, String result) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("-- COMPOST MONITOR --");
  display.setCursor(0, 16);
  display.printf("Temp    : %.1f C\n", temp);
  display.printf("Moisture: %.0f\n", moisture);
  display.printf("Methane : %.0f\n", methane);
  display.setCursor(0, 48);
  display.println("Status:");
  display.setCursor(0, 56);
  if      (result == "COMPOST_READY") display.println(">> READY!");
  else if (result == "TOO_DRY")       display.println(">> TOO DRY");
  else if (result == "TOO_WET")       display.println(">> TOO WET");
  else                                display.println(result);
  display.display();
}

void showMessage(const char* line1, const char* line2, const char* line3) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 10); display.println(line1);
  display.setCursor(0, 26); display.println(line2);
  display.setCursor(0, 42); display.println(line3);
  display.display();
}

void sendToCloud(float temp, float moisture, float methane, String result) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  String body = "{\"temperature\":" + String(temp, 2) +
                ",\"moisture\":"    + String(moisture, 0) +
                ",\"methane\":"     + String(methane, 0) +
                ",\"result\":\""    + result + "\"}";

  int code = http.POST(body);
  Serial.printf("Cloud POST: %d\n", code);
  http.end();
}

void checkCommand() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(COMMAND_URL);
  int code = http.GET();

  if (code == 200) {
    String response = http.getString();
    StaticJsonDocument<128> doc;
    deserializeJson(doc, response);
    String cmd = doc["command"].as<String>();

    if (cmd == "reset") {
      Serial.println("Command: reset");
      showMessage("CMD:", "RESET", "Restarting...");
      delay(1000);
      ESP.restart();
    } else if (cmd == "status") {
      Serial.println("Command: status");
    }
  }
  http.end();
}
