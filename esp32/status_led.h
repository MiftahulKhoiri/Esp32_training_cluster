// status_led.h — indikator LED status kerja ESP32 (versi ESP32/Arduino)
#pragma once
#include <Arduino.h>

class StatusLed {
public:
    // LED bawaan ESP32 DevKit V1 umumnya di GPIO2. Kalau boardmu beda,
    // ganti LED_PIN sesuai datasheet/pinout board masing-masing.
    static const int LED_PIN = 2;

    static void init() {
        pinMode(LED_PIN, OUTPUT);
        digitalWrite(LED_PIN, LOW);
    }

    // Nyalakan LED — dipanggil di awal setiap tahap kerja (connect wifi,
    // receive weight, training, send weight)
    static void working() {
        digitalWrite(LED_PIN, HIGH);
    }

    // Matikan LED — dipanggil saat node idle/nunggu ronde berikutnya
    static void idle() {
        digitalWrite(LED_PIN, LOW);
    }

    // Kedip cepat beberapa kali untuk tandai error (WiFi gagal, HTTP gagal, dll).
    // Blocking sebentar — dipakai cuma saat error, bukan di jalur kerja normal.
    static void blink_error(int times = 5) {
        for (int i = 0; i < times; ++i) {
            digitalWrite(LED_PIN, HIGH);
            delay(100);
            digitalWrite(LED_PIN, LOW);
            delay(100);
        }
    }
};