// status_led.h — indikator LED status kerja ESP32 (versi ESP32/Arduino)
#pragma once
#include <Arduino.h>

class StatusLed {
public:
    static const int LED_PIN = 2;

    static void init() {
        pinMode(LED_PIN, OUTPUT);
        digitalWrite(LED_PIN, LOW);
    }

    static void working() {
        digitalWrite(LED_PIN, HIGH);
    }

    static void idle() {
        digitalWrite(LED_PIN, LOW);
    }

    static void blink_error(int times = 5) {
        for (int i = 0; i < times; ++i) {
            digitalWrite(LED_PIN, HIGH);
            delay(100);
            digitalWrite(LED_PIN, LOW);
            delay(100);
        }
    }

    // Pola beda dari error: kedip lambat 3x (dipanggil sekali saat satu
    // tugas komputasi selesai), biar gampang dibedakan dari blink_error yang cepat.
    static void task_done() {
        for (int i = 0; i < 3; ++i) {
            digitalWrite(LED_PIN, HIGH);
            delay(500);
            digitalWrite(LED_PIN, LOW);
            delay(500);
        }
    }
};